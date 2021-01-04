import os
import math
import json
import cv2

import numpy as np
import pprint as pp
import json as JSON
import pandas as pd
import pandas_gbq

from shapely.geometry import Point, Polygon
from dotenv import load_dotenv
from matplotlib import pyplot as plt

from OSMPythonTools.api import Api
# Tool to search OSM data by name and address
from OSMPythonTools.nominatim import Nominatim
# read-only API that serves up custom selected parts of the OSM map data
from OSMPythonTools.overpass import overpassQueryBuilder, Overpass


class BigQueryHelper:

    def __init__(self):
        load_dotenv()

    # ___MAPS IMAGES___

    def saveImages(self, images_df: pd.DataFrame) -> None:
        """ Replace all Table with DataFrame

        Args:
            images_df (DataFrame): DataFrame with all images

         Returns:
             None
         """

        destination_table = 'osm_data.maps_images'
        pandas_gbq.to_gbq(images_df, project_id='sidhouses',
                          destination_table=destination_table, if_exists='replace')

    def loadImages(self) -> pd.DataFrame:
        """Query All the images from the BigQuery Table

         Returns:
             Returns a DataFrame with all the images
         """

        df = pandas_gbq.read_gbq(
            'SELECT * FROM `sidhouses.osm_data.maps_images`', project_id='sidhouses')
        return df

    # ___SID BUILDINGS___

    def saveSIDBuildings(self, buildings_df: pd.DataFrame):
        """ Replace all Table with DataFrame

        Args:
            buildings_df (DataFrame): DataFrame with all detected Buildings

         Returns:
             None
        """

        destination_table = 'output_data.buildings'
        buildings_df = buildings_df.drop(columns=['polygon', 'point'], axis=1)
        pandas_gbq.to_gbq(buildings_df, project_id='sidhouses',
                          destination_table=destination_table, if_exists='replace')

    def loadSIDBuildings(self) -> pd.DataFrame:
        """Query All the buildings from the BigQuery Table
            Generating the Polygons and Points

         Returns:
             Returns a DataFrame with all the buildings
         """

        df = pandas_gbq.read_gbq(
            'SELECT * FROM sidhouses.output_data.buildings', project_id='sidhouses')
        self.create_polygons(df)
        self.create_points(df)
        return df

    # ___OSM BUILDINGS___

    def saveOSMBuildings(self, buildings_df: pd.DataFrame):
        """ Replace all Table with DataFrame

        Args:
            buildings_df (DataFrame): DataFrame with all Buildings

         Returns:
             None
        """

        destination_table = 'osm_data.buildings'
        buildings_df = buildings_df.drop(columns=['polygon', 'point'], axis=1)
        pandas_gbq.to_gbq(buildings_df, project_id='sidhouses',
                          destination_table=destination_table, if_exists='replace')

    def loadOSMBuildings(self) -> pd.DataFrame:
        """Query All the buildings from the BigQuery Table
            Generating the Polygons and Points

         Returns:
             Returns a DataFrame with all the buildings
         """

        df = pandas_gbq.read_gbq('SELECT * FROM sidhouses.osm_data.buildings', project_id='sidhouses')
        self.create_polygons(df)
        self.create_points(df)
        return df

    # ___DISTRICTS___

    def saveDistricts(self, districts_df: pd.DataFrame):
        """ Replace all Table with DataFrame

        Args:
            districts_df (DataFrame): DataFrame with all Districts

         Returns:
             None
        """

        destination_table = 'osm_data.districts'
        districts_df = districts_df.drop(columns=['polygon', 'point'], axis=1)
        pandas_gbq.to_gbq(districts_df, project_id='sidhouses',
                          destination_table=destination_table, if_exists='replace')

    def loadDistricts(self) -> pd.DataFrame:
        """Query All the districts from the BigQuery Table
            Generating the Polygons and Points

         Returns:
             Returns a DataFrame with all the districts
         """

        df = pandas_gbq.read_gbq('SELECT * FROM sidhouses.osm_data.districts')
        self.create_polygons(df)
        self.create_points(df)
        return df

    # ___VILLAGES___

    def saveVillages(self, village_df: pd.DataFrame) -> None:
        """ Replace all Table with DataFrame

        Args:
            village_df (DataFrame): DataFrame with all villages

         Returns:
             None
         """

        destination_table = 'osm_data.villages'
        village_df = village_df.drop(columns=['polygon', 'point'], axis=1)
        pandas_gbq.to_gbq(village_df, project_id='sidhouses',
                          destination_table=destination_table, if_exists='replace')

    def loadVillages(self) -> pd.DataFrame:
        """Query All the villages from the BigQuery Table
            Generating the Polygons and Points

         Returns:
             Returns a DataFrame with all the villages
         """

        df = pandas_gbq.read_gbq('SELECT * FROM sidhouses.osm_data.villages')
        self.create_polygons(df)
        self.create_points(df)
        return df

    # --------- AUX ---------

    def create_polygons(self, df: pd.DataFrame) -> None:
        """Check for column name 'boundary_lon' and 'boundary_lat' and generate column with Polygons
        ´
        Args:
            df (DataFrame): DataFrame with columns 'boundary_lon' and 'boundary_lat'

         Returns:
             None
         """

        df['polygon'] = ""
        for index, row in df.iterrows():
            if (row['boundary_lon'] != '' and row['boundary_lat'] != '' and row['boundary_lat'].isnull() == False and row['boundary_lat'].isnull() == False):
                lon = json.loads(row['boundary_lon'])
                lat = json.loads(row['boundary_lat'])
                df.at[index, 'polygon'] = Polygon(zip(lon, lat))

    def create_points(self, df: pd.DataFrame) -> None:
        """Check for column name 'lat' and 'lon' and generate column with Points

        Args:
            df (DataFrame): DataFrame with columns 'lat' and 'lon'

         Returns:
             None
        """

        df['point'] = ""
        for index, row in df.iterrows():
            if (row['lat'] != '' and row['lon'] != ''):
                p = Point(float(row['lon']), float(row['lat']))
                df.at[index, 'point'] = p


class OSMQueryHelper:

    api = Api()
    nominatim = Nominatim()
    overpass = Overpass()

    # AREA ID
    BENIN_ID = 3600192784
    BORGOU_ID = 3602803880
    NIKKI_COM_ID = 3602859963
    NIKKI_ARR_ID = 3611176398

    # RELATION ID
    BENIN_ID_REL = 192784
    BORGOU_ID_REL = 2803880
    NIKKI_ID_REL = 2859963

    # CONSTANTS
    DEG_to_KM2 = 13000
    DEG_to_KM = 130

    def __init__(self):
        None

    def getAreaInOSM(self, osm_id, type: str) -> int:
        if type == 'way':
            return int(osm_id) + 2400000000
        if type == 'relation':
            return int(osm_id) + 3600000000
        return None

    def updateAreaIdOfDataframe(self, df: pd.DataFrame) -> None:
        for index, row in df.iterrows():
            if row['area_id'] == '' or int(row['area_id']) == 0:
                df.at[index, 'area_id'] = self.getAreaInOSM(
                    row['osm_id'], row['type'])

    def countBuildingsInAreaID(self, area_id: int) -> int:
        buildings_query = overpassQueryBuilder(
            area=area_id, elementType=['way'], selector="building", out='count')
        buildings_res = self.overpass.query(buildings_query)
        return buildings_res.countWays()

    def updateBuildingsInVillage(self, village_df: pd.DataFrame) -> None:
        for index, village_row in village_df.iterrows():
            nb_buildings = 0
            density = 0

            # Check if there have already been registered
            if int(village_row['nb_buildings']) < 0:
                nb_buildings = self.countBuildingsInAreaID(
                    village_row['area_id'])

            # Update the density
            if nb_buildings > 0:
                density = nb_buildings / \
                    (float(str(village_row['area']).replace(
                        ',', '.')) * 1000000.0)

                village_df.loc[index, 'nb_buildings'] = nb_buildings
                village_df.loc[index, 'build_dens'] = density
                village_df.loc[index, 'pop_est'] = nb_buildings * 5

    def isIdalready(self, df: pd.DataFrame, _id: int) -> bool:
        for row in df.iterrows():
            if int(row['osm_id']) == int(_id):
                return True
        return False

    def updateDistrictOfVillage(self, district_df: pd.DataFrame, village_df: pd.DataFrame) -> None:
        # Loop throught all villages
        for i, village_row in village_df.iterrows():

            # Check if there is not already a district
            if village_row['district'] == '':

                # Loop throught all districts to see if the village is inside the polygon
                for district_row in district_df.iterrows():
                    point = village_row['point']
                    isin = district_row['polygon'].contains(point)
                    if isin == True:
                        village_df.at[i, 'district'] = district_row['name']
                        break

    def getVillagesFromOSM(self) -> pd.DataFrame:
        # Build the query
        village_query = overpassQueryBuilder(
            area=self.NIKKI_COM_ID, elementType='way', selector="'place'~'village|locality|town|city|hamlet'", out="body geom")
        residential_query = overpassQueryBuilder(
            area=self.NIKKI_COM_ID, elementType='way', selector="'landuse'~'residential'", out="body geom")

        # Make the query
        village_res = self.overpass.query(village_query)
        residential_res = self.overpass.query(residential_query)

        # Transform response into json
        village_json = village_res.toJSON()
        residential_res = residential_res.toJSON()

        village_headers = ['osm_id', 'area_id', 'type', 'place', 'district', 'name', 'link', 'lat', 'lon', 'perim', 'area',
                           'boundary_lat', 'boundary_lon', 'polygon', 'point', 'nb_buildings', 'building_density', 'pop_est']
        new_village_data = []

        # Loop throught the elements in the response
        for ele in village_json['elements']:
            geo_df = pd.DataFrame.from_records(ele['geometry'])
            poly = Polygon(zip(geo_df['lon'], geo_df['lat']))
            point = Point(poly.centroid.x, poly.centroid.y)
            boudary_lat = str(geo_df['lat'].tolist())
            boudary_lon = str(geo_df['lon'].tolist())
            ele_row = [
                ele.get('id', 0),                    # osm_id
                0,                                  # area_id
                ele.get('type', ''),                 # type
                ele['tags'].get('place', ''),        # place
                '',                                 # district
                ele['tags'].get('name', ''),         # name
                'https://www.openstreetmap.org/edit?' + \
                str(ele.get('type', '')) + '=' + \
                str(ele.get('id', '')),  # link
                str(poly.centroid.y),               # lat
                str(poly.centroid.x),               # lon
                str(poly.length * self.DEG_to_KM),       # perim
                str(poly.area * self.DEG_to_KM2),        # area
                boudary_lat,                        # boudary_lat
                boudary_lon,                        # boudary_lon
                poly,                               # polygon
                point,                              # point
                -1,                                 # nb_buildings
                0.0,                                # building_density
                0.0,                                # pop_est
            ]

            new_village_data.append(ele_row)

        # Loop throught the elements in the response
        for ele in residential_res['elements']:
            geo_df = pd.DataFrame.from_records(ele['geometry'])
            poly = Polygon(zip(geo_df['lon'], geo_df['lat']))
            point = Point(poly.centroid.x, poly.centroid.y)
            boudary_lat = str(geo_df['lat'].tolist())
            boudary_lon = str(geo_df['lon'].tolist())
            ele_row = [
                ele.get('id', 0),                    # osm_id
                0,                                  # area_id
                ele.get('type', ''),                 # type
                ele['tags'].get('landuse', ''),      # landuse
                '',                                 # district
                ele['tags'].get('name', ''),         # name
                'https://www.openstreetmap.org/edit?' + \
                str(ele.get('type', '')) + '=' + \
                str(ele.get('id', '')),  # link
                str(poly.centroid.y),               # lat
                str(poly.centroid.x),               # lon
                str(poly.length * self.DEG_to_KM),       # perim
                str(poly.area * self.DEG_to_KM2),        # area
                boudary_lat,                        # boudary_lat
                boudary_lon,                        # boudary_lon
                poly,                               # polygon
                point,                              # point
                -1,                                 # nb_buildings
                0.0,                                # building_density
                0.0,                                # population_est_building
            ]

            new_village_data.append(ele_row)

        # Create Dataframe from data
        new_village_df = pd.DataFrame(
            new_village_data, columns=village_headers)
        return new_village_df

    def getBuildingsInVillage(self, village_row_df: pd.DataFrame) -> any:

        # Build the query
        building_query = overpassQueryBuilder(
            area=village_row_df['area_id'].values[0], elementType='way', selector="building", out="geom")

        # Make the query
        building_res = self.overpass.query(building_query)

        # Transform response into json
        building_json = building_res.toJSON()
        new_building_data = []

        # Loop throught the elements in the response
        for ele in building_json['elements']:
            geo_df = pd.DataFrame.from_records(ele['geometry'])
            poly = Polygon(zip(geo_df['lon'], geo_df['lat']))
            point = Point(poly.centroid.x, poly.centroid.y)
            boudary_lat = str(geo_df['lat'].tolist())
            boudary_lon = str(geo_df['lon'].tolist())
            ele_row = [
                ele.get('id', 0),                    # osm_id
                ele.get('type', ''),                 # type
                village_row_df['district'].values[0],         # district
                village_row_df['name'].values[0],             # village
                str(poly.centroid.y),               # lat
                str(poly.centroid.x),               # lon
                str(poly.length * self.DEG_to_KM),       # perim
                str(poly.area * self.DEG_to_KM2),        # area
                boudary_lat,                        # boudary_lat
                boudary_lon,                        # boudary_lon
                poly,                               # polygon
                point,                              # point
            ]

            new_building_data.append(ele_row)
        return new_building_data


class ConversionHelper:

    def __init__(self):
        None

    def point_px_to_deg(self, lat: int, lon: int, row_ref: pd.DataFrame) -> (float, float):
        """Transforms coords in px to lat,lon in deg

        Args:
            lat (int): latitud in Pixels
            lon (int): longitud in Pixels
            row_ref (DataFrame): DataFrame Row from the corresponding image

         Returns:
            lat_deg (float): latitud in Deg
            lon_deg (float): longitud in Deg
        """

        deg_per_px_lat = row_ref.iloc[0]['deg_per_px_lat']
        deg_per_px_lon = row_ref.iloc[0]['deg_per_px_lon']

        lat = lat + 43 * math.floor(lat / row_ref.iloc[0]['input_size'])

        lat_deg = row_ref.iloc[0]['lat_0'] - lat * deg_per_px_lat
        lon_deg = row_ref.iloc[0]['lon_0'] + lon * deg_per_px_lon

        return lat_deg, lon_deg

    def point_deg_to_px(self, lat: float, lon: float, row_ref: pd.DataFrame) -> (int, int):
        """Transforms coords in deg to lat,lon in px

        Args:
            lat (float): latitud in Deg
            lon (float): longitud in Deg
            row_ref (DataFrame): DataFrame Row from the corresponding image

         Returns:
            lat_px (int): latitud in Px
            lon_px (int): longitud in Px
        """

        deg_per_px_lat = row_ref.iloc[0]['deg_per_px_lat']
        deg_per_px_lon = row_ref.iloc[0]['deg_per_px_lon']

        lat_px = (row_ref.iloc[0]['lat_0'] - lat) / deg_per_px_lat
        lon_px = (lon - row_ref.iloc[0]['lon_0']) / deg_per_px_lon

        lat_px = lat_px - 43 * \
            math.floor(lat_px / row_ref.iloc[0]['input_size'])
        return lat_px, lon_px


class TestingHelper:

    def __init__(self):
        None

    def show_point_in_img(self, cnv: ConversionHelper, lat: float, lon: float, image_ref: str,  row_ref: any) -> (int, int):

        img = cv2.imread(image_ref)
        fig, ax = plt.subplots(figsize=(15, 185))
        lat_px, lon_px = cnv.point_deg_to_px(lat, lon, row_ref)

        point = plt.Circle((lon_px, lat_px), 5, color='b')
        ax.add_artist(point)

        ax.imshow(img)

        return lat_px, lon_px

    def show_point_in_img_to_deg(self, cnv: ConversionHelper, lat: int, lon: int, image_ref: str,  row_ref: any) -> (float, float):

        img = cv2.imread(image_ref)
        fig, ax = plt.subplots(figsize=(15, 185))
        lat_deg, lon_deg = cnv.point_px_to_deg(lat, lon, row_ref)

        point = plt.Circle((lon, lat), 5, color='b')
        ax.add_artist(point)

        ax.imshow(img)

        return lat_deg, lon_deg


class MapsImage:
    """ Image in BigQuery Table

    Args:

        img_name (str): image file name with .png
        name (str): village name

        input_lat (float): lat in deg entered to get the image
        input_lon (float): lon in deg entered to get the image
        input_size (int): image size entered to get the image
        grid_size (int): grid size entered to get the image

        lat_size (float): img vertical size in px 
        lon_size (float): img horizontal size in px 

        lat_0 (float): lat in deg from the top left corner
        lon_0 (float): lon in deg from the top left corner

        m_per_px_lat (float): meters per pixel in latitud
        m_per_px_lon (float): meters per pixel in longitud

        deg_per_m_lat (float): deg per meter in latitud
        deg_per_m_lat (float): deg per meter in longitud

        deg_per_px_lat (float): deg per px in latitud
        deg_per_px_lat (float): deg per px in longitud

    """
    img_name: str
    name: str

    input_lat: float
    input_lon: float
    input_size: int
    grid_size: int

    lat_size: int
    lon_size: int

    lat_0: float
    lon_0: float

    m_per_px_lat: float
    m_per_px_lon: float

    deg_per_m_lat: float
    deg_per_m_lon: float

    deg_per_px_lat: float
    deg_per_px_lon: float


class District:
    """ District in BigQuery Table

    Args:

        osm_id (int): OSM id
        area_id (int): OSM area id

        type (str): OSM element type [way, relation, node]
        name (str): district name
        link (str): OSM edit link

        lat (float): lat in deg from the center
        lon (float): lon in deg from  the center

        perim (float): perimeter in km  
        area (float): area in km2

        boundary_lat(str): list of lat coords from the boundary
        boundary_lon(str): list of lon coords from the boundary

        nb_villages (int): Number of villages inside village area
        population (float): Estimation of the population
        density (float): Number of villages / area

        polygon (Polygon): Polygon of the shape
        point (Point): Point in the center

    """
    osm_id: int
    area_id: int

    type: str
    name: str
    link: str

    lat: float
    lon: float

    perim: float
    area: float

    boundary_lat: str
    boundary_lon: str

    nb_villages: int
    population: float
    density: float

    polygon: Polygon
    point: Point


class Village:
    """ Village in BigQuery Table

    Args:

        osm_id (int): OSM id
        area_id (int): OSM area id

        type (str): OSM element type [way, relation, node]
        place (str): OSM place tag [village, residential]
        district (str): district it belongs to [Biro, Nikki] 
        name (str): village name
        link (str): OSM edit link

        lat (float): lat in deg from the center
        lon (float): lon in deg from  the center

        perim (float): perimeter in km  
        area (float): area in km2

        boundary_lat(str): list of lat coords from the boundary
        boundary_lon(str): list of lon coords from the boundary

        nb_buildings (int): Number of buildings inside village area
        pop_est (float): Estimation of the population
        building_density (float): Number of buildings / area

        polygon (Polygon): Polygon of the shape
        point (Point): Point in the center

    """
    osm_id: int
    area_id: int

    type: str
    place: str
    district: str
    name: str
    link: str

    lat: float
    lon: float

    perim: float
    area: float

    boundary_lat: str
    boundary_lon: str

    nb_buildings: int
    pop_est: float
    building_density: float

    polygon: Polygon
    point: Point


class OSMBuilding:
    """ Building in OSM BigQuery Table

    Args:

        osm_id (int): OSM id

        type (str): OSM element type [way, relation, node]
        district (str): district it belongs to [Biro, Nikki] 
        village (str): village it belongs to 

        lat (float): lat in deg from the center
        lon (float): lon in deg from  the center

        perim (float): perimeter in km  
        area (float): area in km2

        boundary_lat(str): list of lat coords from the boundary
        boundary_lon(str): list of lon coords from the boundary

        polygon (Polygon): Polygon of the shape
        point (Point): Point in the center

    """
    osm_id: int

    type: str
    district: str
    village: str

    lat: float
    lon: float

    perim: float
    area: float

    boundary_lat: str
    boundary_lon: str

    polygon: Polygon
    point: Point


class SIDBuilding:
    """ Building in BigQuery Table

    Args:
        district (str): district it belongs to [Biro, Nikki] 
        village (str): village it belongs to 

        analysis_method (str): analysis detection method [ml,edge,color]

        lat (float): lat in deg from the center
        lon (float): lon in deg from  the center

        perim (float): perimeter in km  
        area (float): area in km2

        boundary_lat(str): list of lat coords from the boundary
        boundary_lon(str): list of lon coords from the boundary

        polygon (Polygon): Polygon of the shape
        point (Point): Point in the center

    """
    type: str
    district: str
    village: str

    analysis_method: str

    lat: float
    lon: float

    perim: float
    area: float

    boundary_lat: str
    boundary_lon: str

    polygon: Polygon
    point: Point
