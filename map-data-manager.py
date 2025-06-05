# map_data_manager.py - Railway Map Data Manager for Interactive Maps
import asyncio
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
from pathlib import Path
import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString, Polygon
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import re

from config import settings

logger = logging.getLogger(__name__)

class RailwayMapDataManager:
    """
    Manages railway map data for interactive visualization
    Supports UK railway regions, lines, stations, and project locations
    """
    
    def __init__(self):
        self.data_dir = Path("map_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # UK Railway regions (from Network Rail structure)
        self.uk_railway_regions = {
            "london_north_eastern": {
                "name": "London North Eastern",
                "code": "LNE",
                "description": "East Coast Main Line, West Highland Line, and cross-Pennine routes",
                "major_cities": ["London", "Edinburgh", "Glasgow", "Newcastle", "Leeds", "York"],
                "color": "#1f77b4"
            },
            "london_north_western": {
                "name": "London North Western", 
                "code": "LNW",
                "description": "West Coast Main Line and cross-country routes",
                "major_cities": ["London", "Birmingham", "Manchester", "Liverpool", "Preston", "Carlisle"],
                "color": "#ff7f0e"
            },
            "western": {
                "name": "Western",
                "code": "WR", 
                "description": "Great Western Main Line and Welsh routes",
                "major_cities": ["London", "Bristol", "Cardiff", "Swansea", "Plymouth", "Exeter"],
                "color": "#2ca02c"
            },
            "southern": {
                "name": "Southern",
                "code": "SR",
                "description": "South Coast routes and London commuter lines",
                "major_cities": ["London", "Brighton", "Portsmouth", "Southampton", "Dover", "Hastings"],
                "color": "#d62728"
            },
            "eastern": {
                "name": "Eastern",
                "code": "ER",
                "description": "East Anglia and East Coast routes",
                "major_cities": ["London", "Cambridge", "Norwich", "Ipswich", "Peterborough", "Kings Lynn"],
                "color": "#9467bd"
            },
            "scotland": {
                "name": "Scotland",
                "code": "SC",
                "description": "Scottish railway network including Highland lines",
                "major_cities": ["Glasgow", "Edinburgh", "Aberdeen", "Dundee", "Inverness", "Stirling"],
                "color": "#8c564b"
            }
        }
        
        # Initialize geocoder
        self.geocoder = Nominatim(user_agent="strategy-ai-railway-mapper")
        
        logger.info("Initialized railway map data manager")

    async def get_railway_regions_geojson(self) -> Dict[str, Any]:
        """Get railway regions as GeoJSON for map visualization"""
        try:
            # Check if cached data exists
            cache_file = self.data_dir / "railway_regions.geojson"
            
            if cache_file.exists():
                logger.info("Loading cached railway regions data")
                with open(cache_file, 'r') as f:
                    return json.load(f)
            
            # Generate regions data
            logger.info("Generating railway regions GeoJSON data")
            
            features = []
            
            for region_id, region_data in self.uk_railway_regions.items():
                # Create approximate region boundaries
                # In production, you'd use actual Network Rail boundary data
                region_boundary = await self._create_region_boundary(region_data)
                
                if region_boundary:
                    features.append({
                        "type": "Feature",
                        "properties": {
                            "region_id": region_id,
                            "name": region_data["name"],
                            "code": region_data["code"],
                            "description": region_data["description"],
                            "color": region_data["color"],
                            "major_cities": region_data["major_cities"]
                        },
                        "geometry": region_boundary
                    })
            
            geojson_data = {
                "type": "FeatureCollection",
                "features": features
            }
            
            # Cache the data
            with open(cache_file, 'w') as f:
                json.dump(geojson_data, f, indent=2)
            
            logger.info(f"Generated GeoJSON data for {len(features)} railway regions")
            return geojson_data
            
        except Exception as e:
            logger.error(f"Error getting railway regions GeoJSON: {e}")
            return {"type": "FeatureCollection", "features": []}

    async def get_railway_lines_geojson(self) -> Dict[str, Any]:
        """Get major railway lines as GeoJSON"""
        try:
            cache_file = self.data_dir / "railway_lines.geojson"
            
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    return json.load(f)
            
            # Define major UK railway lines
            major_lines = [
                {
                    "name": "East Coast Main Line",
                    "description": "London to Edinburgh via York and Newcastle",
                    "stations": ["London King's Cross", "Peterborough", "York", "Newcastle", "Edinburgh"],
                    "region": "london_north_eastern",
                    "color": "#1f77b4"
                },
                {
                    "name": "West Coast Main Line", 
                    "description": "London to Glasgow via Birmingham and Preston",
                    "stations": ["London Euston", "Birmingham", "Manchester", "Preston", "Glasgow"],
                    "region": "london_north_western",
                    "color": "#ff7f0e"
                },
                {
                    "name": "Great Western Main Line",
                    "description": "London to Bristol and South Wales",
                    "stations": ["London Paddington", "Reading", "Bristol", "Cardiff"],
                    "region": "western", 
                    "color": "#2ca02c"
                },
                {
                    "name": "Brighton Main Line",
                    "description": "London to Brighton and South Coast",
                    "stations": ["London Victoria", "Gatwick Airport", "Brighton"],
                    "region": "southern",
                    "color": "#d62728"
                }
            ]
            
            features = []
            
            for line in major_lines:
                # Get coordinates for stations along the line
                line_coords = await self._get_line_coordinates(line["stations"])
                
                if line_coords:
                    features.append({
                        "type": "Feature",
                        "properties": {
                            "name": line["name"],
                            "description": line["description"],
                            "region": line["region"],
                            "color": line["color"],
                            "stations": line["stations"]
                        },
                        "geometry": {
                            "type": "LineString",
                            "coordinates": line_coords
                        }
                    })
            
            geojson_data = {
                "type": "FeatureCollection", 
                "features": features
            }
            
            # Cache the data
            with open(cache_file, 'w') as f:
                json.dump(geojson_data, f, indent=2)
            
            logger.info(f"Generated GeoJSON data for {len(features)} railway lines")
            return geojson_data
            
        except Exception as e:
            logger.error(f"Error getting railway lines GeoJSON: {e}")
            return {"type": "FeatureCollection", "features": []}

    async def get_stations_geojson(self, region: Optional[str] = None) -> Dict[str, Any]:
        """Get railway stations as GeoJSON points"""
        try:
            cache_file = self.data_dir / f"stations_{region or 'all'}.geojson"
            
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    return json.load(f)
            
            # Major UK railway stations
            major_stations = [
                {"name": "London King's Cross", "city": "London", "region": "london_north_eastern", "type": "terminus"},
                {"name": "London Euston", "city": "London", "region": "london_north_western", "type": "terminus"},
                {"name": "London Paddington", "city": "London", "region": "western", "type": "terminus"},
                {"name": "London Victoria", "city": "London", "region": "southern", "type": "terminus"},
                {"name": "Birmingham New Street", "city": "Birmingham", "region": "london_north_western", "type": "interchange"},
                {"name": "Manchester Piccadilly", "city": "Manchester", "region": "london_north_western", "type": "interchange"},
                {"name": "Edinburgh Waverley", "city": "Edinburgh", "region": "scotland", "type": "terminus"},
                {"name": "Glasgow Central", "city": "Glasgow", "region": "scotland", "type": "terminus"},
                {"name": "Cardiff Central", "city": "Cardiff", "region": "western", "type": "terminus"},
                {"name": "Bristol Temple Meads", "city": "Bristol", "region": "western", "type": "interchange"},
                {"name": "York", "city": "York", "region": "london_north_eastern", "type": "interchange"},
                {"name": "Newcastle Central", "city": "Newcastle", "region": "london_north_eastern", "type": "interchange"}
            ]
            
            # Filter by region if specified
            if region:
                major_stations = [s for s in major_stations if s["region"] == region]
            
            features = []
            
            for station in major_stations:
                # Get coordinates for station
                coords = await self._geocode_location(f"{station['name']} railway station, UK")
                
                if coords:
                    features.append({
                        "type": "Feature",
                        "properties": {
                            "name": station["name"],
                            "city": station["city"],
                            "region": station["region"],
                            "station_type": station["type"],
                            "marker_color": self.uk_railway_regions[station["region"]]["color"]
                        },
                        "geometry": {
                            "type": "Point",
                            "coordinates": [coords[1], coords[0]]  # [lng, lat] for GeoJSON
                        }
                    })
            
            geojson_data = {
                "type": "FeatureCollection",
                "features": features
            }
            
            # Cache the data
            with open(cache_file, 'w') as f:
                json.dump(geojson_data, f, indent=2)
            
            logger.info(f"Generated GeoJSON data for {len(features)} stations")
            return geojson_data
            
        except Exception as e:
            logger.error(f"Error getting stations GeoJSON: {e}")
            return {"type": "FeatureCollection", "features": []}

    async def get_projects_by_region(self, region_id: str) -> List[Dict[str, Any]]:
        """Get projects associated with a specific railway region"""
        try:
            # This would integrate with your document database
            # For now, return example project data
            
            example_projects = {
                "london_north_eastern": [
                    {
                        "id": "proj_001",
                        "name": "East Coast Digital Signalling",
                        "description": "Modernization of signalling systems on East Coast Main Line",
                        "status": "In Progress",
                        "location": "York",
                        "coordinates": [53.9584, -1.0803],
                        "documents_count": 15,
                        "completion": 65
                    },
                    {
                        "id": "proj_002", 
                        "name": "Newcastle Station Upgrade",
                        "description": "Platform extensions and accessibility improvements",
                        "status": "Planning",
                        "location": "Newcastle",
                        "coordinates": [54.9783, -1.6178],
                        "documents_count": 8,
                        "completion": 20
                    }
                ],
                "western": [
                    {
                        "id": "proj_003",
                        "name": "Great Western Electrification",
                        "description": "Electrification of Bristol to Cardiff route",
                        "status": "Completed",
                        "location": "Bristol",
                        "coordinates": [51.4545, -2.5879],
                        "documents_count": 32,
                        "completion": 100
                    }
                ]
            }
            
            return example_projects.get(region_id, [])
            
        except Exception as e:
            logger.error(f"Error getting projects for region {region_id}: {e}")
            return []

    async def search_locations(self, query: str) -> List[Dict[str, Any]]:
        """Search for railway-related locations"""
        try:
            results = []
            
            # Search in station names
            for station_data in await self._get_all_stations():
                if query.lower() in station_data["name"].lower():
                    coords = await self._geocode_location(f"{station_data['name']} railway station, UK")
                    if coords:
                        results.append({
                            "name": station_data["name"],
                            "type": "station",
                            "region": station_data["region"],
                            "coordinates": [coords[1], coords[0]],  # [lng, lat]
                            "relevance": self._calculate_relevance(query, station_data["name"])
                        })
            
            # Search in city names
            for region_id, region_data in self.uk_railway_regions.items():
                for city in region_data["major_cities"]:
                    if query.lower() in city.lower():
                        coords = await self._geocode_location(f"{city}, UK")
                        if coords:
                            results.append({
                                "name": city,
                                "type": "city",
                                "region": region_id,
                                "coordinates": [coords[1], coords[0]],
                                "relevance": self._calculate_relevance(query, city)
                            })
            
            # Sort by relevance
            results.sort(key=lambda x: x["relevance"], reverse=True)
            
            return results[:10]  # Return top 10 results
            
        except Exception as e:
            logger.error(f"Error searching locations: {e}")
            return []

    async def get_route_between_stations(self, start_station: str, end_station: str) -> Optional[Dict[str, Any]]:
        """Get route information between two stations"""
        try:
            # Get coordinates for both stations
            start_coords = await self._geocode_location(f"{start_station} railway station, UK")
    async def get_route_between_stations(self, start_station: str, end_station: str) -> Optional[Dict[str, Any]]:
        """Get route information between two stations"""
        try:
            # Get coordinates for both stations
            start_coords = await self._geocode_location(f"{start_station} railway station, UK")
            end_coords = await self._geocode_location(f"{end_station} railway station, UK")
            
            if not start_coords or not end_coords:
                return None
            
            # Calculate distance
            distance = geodesic(start_coords, end_coords).kilometers
            
            # Create simple route line (in production, use actual railway routing)
            route_line = {
                "type": "LineString",
                "coordinates": [
                    [start_coords[1], start_coords[0]],  # [lng, lat]
                    [end_coords[1], end_coords[0]]
                ]
            }
            
            return {
                "start_station": start_station,
                "end_station": end_station,
                "distance_km": round(distance, 1),
                "route_geometry": route_line,
                "estimated_time_minutes": round(distance * 1.5),  # Rough estimate
                "intermediate_stations": await self._find_intermediate_stations(start_coords, end_coords)
            }
            
        except Exception as e:
            logger.error(f"Error getting route between {start_station} and {end_station}: {e}")
            return None

    # ============================================================================
    # HELPER METHODS
    # ============================================================================

    async def _create_region_boundary(self, region_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create approximate boundary for a railway region"""
        try:
            # Get coordinates for major cities in the region
            city_coords = []
            
            for city in region_data["major_cities"][:4]:  # Use up to 4 cities for boundary
                coords = await self._geocode_location(f"{city}, UK")
                if coords:
                    city_coords.append([coords[1], coords[0]])  # [lng, lat]
            
            if len(city_coords) >= 3:
                # Create a simple polygon boundary
                # In production, you'd use actual administrative boundaries
                return {
                    "type": "Polygon",
                    "coordinates": [city_coords + [city_coords[0]]]  # Close the polygon
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating region boundary: {e}")
            return None

    async def _get_line_coordinates(self, stations: List[str]) -> List[List[float]]:
        """Get coordinates for stations along a railway line"""
        try:
            coordinates = []
            
            for station in stations:
                coords = await self._geocode_location(f"{station} railway station, UK")
                if coords:
                    coordinates.append([coords[1], coords[0]])  # [lng, lat]
                    
                # Add small delay to respect geocoding rate limits
                await asyncio.sleep(0.1)
            
            return coordinates
            
        except Exception as e:
            logger.error(f"Error getting line coordinates: {e}")
            return []

    async def _geocode_location(self, location_name: str) -> Optional[Tuple[float, float]]:
        """Geocode a location name to coordinates"""
        try:
            # Check cache first
            cache_file = self.data_dir / "geocode_cache.json"
            cache = {}
            
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
            
            if location_name in cache:
                coords = cache[location_name]
                return (coords["lat"], coords["lng"])
            
            # Geocode the location
            location = self.geocoder.geocode(location_name)
            
            if location:
                # Cache the result
                cache[location_name] = {
                    "lat": location.latitude,
                    "lng": location.longitude
                }
                
                # Save cache
                with open(cache_file, 'w') as f:
                    json.dump(cache, f, indent=2)
                
                return (location.latitude, location.longitude)
            
            return None
            
        except Exception as e:
            logger.error(f"Error geocoding {location_name}: {e}")
            return None

    async def _get_all_stations(self) -> List[Dict[str, Any]]:
        """Get all station data"""
        # This would be expanded with a comprehensive station database
        return [
            {"name": "London King's Cross", "region": "london_north_eastern"},
            {"name": "London Euston", "region": "london_north_western"},
            {"name": "London Paddington", "region": "western"},
            {"name": "London Victoria", "region": "southern"},
            {"name": "Birmingham New Street", "region": "london_north_western"},
            {"name": "Manchester Piccadilly", "region": "london_north_western"},
            {"name": "Edinburgh Waverley", "region": "scotland"},
            {"name": "Glasgow Central", "region": "scotland"},
            {"name": "Cardiff Central", "region": "western"},
            {"name": "Bristol Temple Meads", "region": "western"},
            {"name": "York", "region": "london_north_eastern"},
            {"name": "Newcastle Central", "region": "london_north_eastern"}
        ]

    def _calculate_relevance(self, query: str, text: str) -> float:
        """Calculate relevance score for search results"""
        query_lower = query.lower()
        text_lower = text.lower()
        
        # Exact match
        if query_lower == text_lower:
            return 1.0
        
        # Starts with query
        if text_lower.startswith(query_lower):
            return 0.9
        
        # Contains query
        if query_lower in text_lower:
            return 0.7
        
        # Fuzzy match (simple)
        common_chars = set(query_lower) & set(text_lower)
        if common_chars:
            return len(common_chars) / max(len(query_lower), len(text_lower))
        
        return 0.0

    async def _find_intermediate_stations(self, start_coords: Tuple[float, float], end_coords: Tuple[float, float]) -> List[str]:
        """Find stations between two points"""
        try:
            intermediate = []
            all_stations = await self._get_all_stations()
            
            for station in all_stations:
                station_coords = await self._geocode_location(f"{station['name']} railway station, UK")
                if station_coords:
                    # Check if station is roughly between start and end points
                    start_to_station = geodesic(start_coords, station_coords).kilometers
                    station_to_end = geodesic(station_coords, end_coords).kilometers
                    direct_distance = geodesic(start_coords, end_coords).kilometers
                    
                    # If station is on a reasonable path between start and end
                    if start_to_station + station_to_end <= direct_distance * 1.2:
                        intermediate.append(station["name"])
            
            return intermediate[:5]  # Return up to 5 intermediate stations
            
        except Exception as e:
            logger.error(f"Error finding intermediate stations: {e}")
            return []

    # ============================================================================
    # DATA MANAGEMENT METHODS
    # ============================================================================

    async def update_map_data(self) -> Dict[str, Any]:
        """Update all map data (regions, lines, stations)"""
        try:
            logger.info("Starting map data update")
            
            # Clear cache files
            cache_files = [
                "railway_regions.geojson",
                "railway_lines.geojson", 
                "stations_all.geojson"
            ]
            
            for cache_file in cache_files:
                file_path = self.data_dir / cache_file
                if file_path.exists():
                    file_path.unlink()
            
            # Regenerate data
            regions_data = await self.get_railway_regions_geojson()
            lines_data = await self.get_railway_lines_geojson()
            stations_data = await self.get_stations_geojson()
            
            return {
                "success": True,
                "updated": {
                    "regions": len(regions_data.get("features", [])),
                    "lines": len(lines_data.get("features", [])),
                    "stations": len(stations_data.get("features", []))
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating map data: {e}")
            return {"success": False, "error": str(e)}

    async def get_map_statistics(self) -> Dict[str, Any]:
        """Get statistics about map data"""
        try:
            regions_data = await self.get_railway_regions_geojson()
            lines_data = await self.get_railway_lines_geojson()
            stations_data = await self.get_stations_geojson()
            
            # Count stations by region
            stations_by_region = {}
            for feature in stations_data.get("features", []):
                region = feature["properties"]["region"]
                stations_by_region[region] = stations_by_region.get(region, 0) + 1
            
            return {
                "regions": {
                    "total": len(regions_data.get("features", [])),
                    "details": [
                        {
                            "id": feature["properties"]["region_id"],
                            "name": feature["properties"]["name"],
                            "stations": stations_by_region.get(feature["properties"]["region_id"], 0)
                        }
                        for feature in regions_data.get("features", [])
                    ]
                },
                "lines": {
                    "total": len(lines_data.get("features", [])),
                    "major_routes": [feature["properties"]["name"] for feature in lines_data.get("features", [])]
                },
                "stations": {
                    "total": len(stations_data.get("features", [])),
                    "by_region": stations_by_region
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting map statistics: {e}")
            return {"error": str(e)}

    async def export_map_data(self, format: str = "geojson") -> Optional[str]:
        """Export all map data in specified format"""
        try:
            if format.lower() == "geojson":
                # Combine all GeoJSON data
                regions = await self.get_railway_regions_geojson()
                lines = await self.get_railway_lines_geojson()
                stations = await self.get_stations_geojson()
                
                combined_data = {
                    "type": "FeatureCollection",
                    "metadata": {
                        "title": "UK Railway Network Map Data",
                        "generated": datetime.now().isoformat(),
                        "source": "Strategy AI Railway Mapper"
                    },
                    "layers": {
                        "regions": regions,
                        "lines": lines,
                        "stations": stations
                    }
                }
                
                # Save combined file
                export_file = self.data_dir / f"uk_railway_network_{datetime.now().strftime('%Y%m%d')}.geojson"
                with open(export_file, 'w') as f:
                    json.dump(combined_data, f, indent=2)
                
                logger.info(f"Exported map data to {export_file}")
                return str(export_file)
            
            return None
            
        except Exception as e:
            logger.error(f"Error exporting map data: {e}")
            return None

    def get_map_config(self) -> Dict[str, Any]:
        """Get configuration for map visualization"""
        return {
            "default_center": [54.7, -2.0],  # Center of UK
            "default_zoom": 6,
            "max_zoom": 18,
            "min_zoom": 5,
            "tile_layers": {
                "default": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                "satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                "transport": "https://{s}.tile.thunderforest.com/transport/{z}/{x}/{y}.png"
            },
            "style_config": {
                "regions": {
                    "fill_opacity": 0.2,
                    "stroke_weight": 2,
                    "stroke_opacity": 0.8
                },
                "lines": {
                    "weight": 3,
                    "opacity": 0.8
                },
                "stations": {
                    "radius": 6,
                    "fill_opacity": 0.8,
                    "stroke_weight": 2
                }
            },
            "interaction": {
                "enable_clustering": True,
                "cluster_distance": 50,
                "popup_max_width": 300,
                "tooltip_enabled": True
            }
        }