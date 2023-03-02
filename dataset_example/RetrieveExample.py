################################################################################
#   This code is an example of how to use the Frost API. The code might        #
#   contain some bugs for different frequencies or trying to obtain data       #
#   that is not available. Nevertheless, the code should (and is only          # 
#   intended) to give a user some experience in how to use the Frost API.      #
#   Remeber to cite our paper if you found any of this useful, as well         #
#   as acknowledging MET Norway for their open-source weather data             #
#   See the following resources for more information:                          #
#       https://frost.met.no/howto.html                                        #
#       https://frost.met.no/api.html#/                                        # 
#       https://www.met.no/en/free-meteorological-data/Licensing-and-crediting #             
#                                                                              #
################################################################################

import matplotlib.pyplot as plt
import requests
import pandas as pd
import utils_geom
import numpy as np
import os
from math import sin, cos, sqrt, atan2, radians, atan, tan


# Insert your own client ID here
client_id = ''


def get_data(endpoint, parameters):
    r = requests.get(endpoint, parameters, auth=(client_id, ''))
    # Extract JSON data
    json = r.json()

    # Check if the request worked, print out any errors
    if r.status_code == 200:
        data = json['data']
        # print('Data retrieved from frost.met.no!')
    else:
        print('Error! Returned status code %s' % r.status_code)
        print('Message: %s' % json['error']['message'])
        print('Reason: %s' % json['error']['reason'])
        raise ValueError

    return data


def get_stations(coord_main, polygon):
    print('Starting get_stations')
    endpoint = 'https://frost.met.no/sources/v0.jsonld'

    polygon_str = str([str(loc).translate({ ord(c): None for c in ",[]" }) for loc in polygon])
    polygon_str = 'POLYGON((' + polygon_str.translate({ ord(c): None for c in "[]''" }) + '))'

    parameters = {
        'geometry': polygon_str
    }
    data = get_data(endpoint, parameters)

    # This will return a Dataframe with all of the observations in a table format
    df = pd.DataFrame()
    for i in range(len(data)):
        row = pd.Series()
        row['name'] = data[i]['name']
        row['id'] = data[i]['id']
        row['validFrom'] = data[i]['validFrom']
        coord_i = data[i]['geometry']['coordinates']
        row['lon'] = coord_i[0]
        row['lat'] = coord_i[-1]

        # Get the distance to the target coordinate:
        dist_tar = utils_geom.distance_coord(coord_main[::-1], coord_i[::-1])
        row['dist_tar'] = dist_tar
        df = df.append(row, ignore_index=True)
    df = df.sort_values('dist_tar').reset_index(drop=True)

    return df


def get_info(df, elements_list):
    print('Starting get_info')
    endpoint = 'https://frost.met.no/observations/availableTimeSeries/v0.jsonld'

    elements = ', '.join(elements_list)

    info_df = pd.DataFrame()    # Temporarily store the variable for each station
    for i, row in df.iterrows():
        parameters = {
            'sources': row['id'],
            'elements': elements
        }
        data = get_data(endpoint, parameters)
        info_i = pd.Series()
        for element_i in elements_list:
            info_i['id'] = row['id']
            if element_i in [d['elementId'] for d in data]:
                data_i = data[[d['elementId'] for d in data].index('wind_speed')]
                info_i[element_i + '_validFrom'] = data_i['validFrom']
                info_i[element_i + '_resolution'] = data_i['timeResolution']
            else:
                info_i[element_i + '_validFrom'] = str(99999)
                info_i[element_i + '_resolution'] = str(99999)
        info_df = info_df.append(info_i, ignore_index=True)

    df = pd.merge(df, info_df, on='id')

    return df, elements_list


def get_timeseries(stations, save_path, end_time='2022-03-01T00:00:00.000Z', missing_val=99999):
    print('Starting get_timeseries')
    endpoint = 'https://frost.met.no/observations/v0.jsonld'

    # Get the measurement elements we want to download:
    resol_cols = stations.columns[stations.columns.str.contains('resolution')]
    elements_list = resol_cols.str.replace('_resolution', '')
    elements = ', '.join(elements_list)

    # Get the valid start time, so that all measurements have started recording
    time_cols = stations.columns[stations.columns.str.contains('validFrom')]
    valid_time = stations[time_cols].values.flatten()
    valid_time.sort()
    valid_time = valid_time[-1]

    # Get the time interval and find how many datapoints it corresponds to (max size per request == 100,000):
    time_interval = pd.to_datetime(end_time) - pd.to_datetime(valid_time)
    total_points = time_interval.total_seconds() / (60*10)   # Assuming res to be in minute resolution
    total_points = total_points*len(elements_list)
    num_queries = int(np.ceil(total_points / 100000))
    time_range = pd.date_range(pd.to_datetime(valid_time), pd.to_datetime(end_time), num_queries + 1)
    time_range = time_range.to_series().dt.strftime('%Y-%m-%d').to_numpy()  # '%Y-%m-%dT%H:%M:%SZ' for iso format
    time_range = np.vstack((time_range[:-1], time_range[1:])).T

    for print_i, (_, row) in enumerate(stations.iterrows()):
        print('   Starting row: ', print_i, ' of ', stations.shape[0])
        station_i = row['name']
        id_i = row['id']
        for print_ii, (s_ii, e_ii) in enumerate(time_range):
            print('      Starting ', print_ii, '/', len(time_range))
            parameters = {
                'sources': id_i,
                'referencetime': s_ii + '/' + e_ii,
                'elements': elements
            }
            try:
                data = get_data(endpoint, parameters)
            except:
                print('***No Valid Data for ', station_i, 'in the period: [', s_ii, ', ', e_ii, ']***')
                continue
            df = pd.DataFrame()
            for d in data:
                df_i = pd.Series([missing_val]*len(elements_list), elements_list)
                df_i['time'] = d['referenceTime']
                for d_i in d['observations']:
                    if 'qualityCode' not in d_i.keys():
                        d_i['qualityCode'] = 0
                    if d_i['qualityCode'] > 1:   #  or d_i['timeResolution'] != 'PT10M':
                        continue
                    df_i[d_i['elementId']] = d_i['value']
                if (missing_val == df_i[df_i.index != 'time']).all():
                    continue
                df = df.append(df_i, ignore_index=True)
            if len(df) == 0:
                print('No valid data for period: ', s_ii, ' / ', e_ii)
            df['id'] = id_i
            df_name = s_ii + '_' + e_ii + '.csv'
            path_i = os.path.join(save_path, station_i.replace(' ', ''), df_name)
            if not os.path.exists(os.path.dirname(path_i)):
                os.makedirs(os.path.dirname(path_i))
            print('Saved file to: ', path_i)
            df.to_csv(path_i, sep=',', index=False)


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    
    # Let's say that you want to find stations that are close to coord_main, but lie within the polygon 
    # specified below. get_stations() will return the available stations in the specified polygon in 
    # sorted order of which are closest to coord_main.
    # This is useful before you start downloading. 
    coord_main = [9.8, 58.1]
    polygon = [
        [9., 57.],
        [9., 59.],
        [11., 59.],
        [11, 57.],
    ]
    
    # Get stations inside "polygon" in sorted order of which are closest to "coord_main"
    stations = get_stations(coord_main, polygon)

    # Select the closest stations to coord_main for example: 
    stations = stations.iloc[:2]

    # Specify some elements you want to inspect, see https://frost.met.no/api.html#/ for which are available: 
    elements_list = ['wind_speed',
                     'wind_from_direction',
                     'air_pressure_at_sea_level',
                     'air_temperature',]
    
    # Get the period for which the stations have the elements available and at what resolution
    # This function might fail if a specified stations doesnt have any available data for the 
    # specified elements (just a trivial example).
    stations, elements_list = get_info(stations, elements_list)

    get_timeseries(stations, save_path='RawData')
