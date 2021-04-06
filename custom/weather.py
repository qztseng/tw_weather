from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import requests
from datetime import date, datetime

from matplotlib import cm
import matplotlib
import folium
import geocoder

matplotlib.rcParams['font.family'] = ['Heiti TC']


def get_Monthly_Weather(station:str, yearmonth:str, session=None) -> pd.DataFrame:
    '''
    get daily weather report of specified station for a given month 
    
    Arguments:
    ----------
    station: 6-character alphanumerical code, 
        can be found at https://e-service.cwb.gov.tw/wdps/obs/state.htm
    yearmonth: date string in the format 'yyyy-mm'
    session: a request.Session to speed up consecutive requests
    
    Returns:
    --------
    a parsed pandas dataframe
    '''
    
    url = 'http://e-service.cwb.gov.tw/HistoryDataQuery/MonthDataController.do'
    
    if session is None:
        session = requests.Session()
    
    try:
        r = session.get(url, **{'params':
                                {'command':'viewMain',
                                 'station':station,
                                 'stname':"",
                                 'datepicker':yearmonth
                                },
                                'verify':False,
                               })
        r.encoding = 'utf8'

        dfs = pd.read_html(r.text, header=0)[1]

    except Exception as e:
        print('Error:', e)

    dfs.columns =  dfs.iloc[1,:]
    dfs.drop(index=[0,1,], inplace=True)
    dfs['date'] = dfs.iloc[:,0].apply(lambda x: datetime.strptime(yearmonth+"-"+x, "%Y-%m-%d"))
    
    return dfs.drop(columns=dfs.columns[0]).set_index(keys='date')

def get_Yearly_Weather(station:str, year:str, month_end=12, verbose=True, session=None) -> pd.DataFrame:
    '''
    Concatenate daily weather data from the get_Monthly_Weather function
    
    Arguments:
    ----------
    station: 6-character alphanumerical code, 
        can be found at https://e-service.cwb.gov.tw/wdps/obs/state.htm
    year: date string in the format 'yyyy'
    session: a request.Session to speed up consecutive requests
    
    Returns:
    --------
    a parsed pandas dataframe
    '''
    if session is None:
        session = requests.Session()
    
    df = pd.DataFrame()
    for m in range(1,month_end+1):
        ym = (year+f'-{m:02}')
        if(verbose):print(ym)
        temp = get_Monthly_Weather(station, ym, session)
        df = df.append(temp)
    
    return df

def get_historical_weather(station:str, y_start:int=2010, y_end:int=date.today().year, verbose=True) ->pd.DataFrame:
    '''
    Concatenate daily weather data from the get_Yearly_Weather function
    
    Arguments:
    ----------
    station: 6-character alphanumerical code, 
        can be found at https://e-service.cwb.gov.tw/wdps/obs/state.htm
    y_start: query starting year, in the format yyyy (int)
    y_end: query ending year, in the format yyyy (int)
    
    Returns:
    --------
    a parsed pandas dataframe
    '''
    session = requests.Session()

    #st = '466920'  #台北466920  宜蘭467080
    his = pd.DataFrame()
    for y in range(y_start, y_end+1):
        yy = f'{y:04}'
        if(verbose):print('Fetching data for year: '+ yy)
        today = date.today()
        m_end=12
        if(y == today.year):
            m_end = today.month
        temp = get_Yearly_Weather(station, yy, m_end, verbose, session)
        his = his.append(temp)

    return his

def cleanup_weather(df):
    
    out = df.copy()
    out.update(out.filter(regex='^Precp.*').replace('T', '0')) 
    #fix the ... value in PrecpHour while zero precipitation
    mask = pd.to_numeric(out['Precp'], errors='coerce')==0
    out.update(out.loc[mask, 'PrecpHour'].replace('...', '0.0'))

    out.drop(columns=out.columns[out.columns.str.match('.+Time')], inplace=True) 
    #discard variables ususally not present in all stations
    col_drop = ['VisbMean', 'EvapA', 'UVI Max','Cloud Amount']
    out.drop(columns=col_drop, inplace=True) 

    #discard potentially highly correlated columns
    col_drop = ['PrecpMax10', 'SunShine', 'SeaPres'] 
    out.drop(columns=col_drop, inplace=True) 
  
    out = out.apply(pd.to_numeric, errors='coerce')

    out['PresDif'] = out.StnPresMax - out.StnPresMin
    out['TempDif'] = out['T Max'] - out['T Min']
    out['RHDif'] = out['RH'] - out['RHMin']
    
    out.dropna(axis=0, how='all', inplace=True)
    
    return out

def impute_ExtraTrees(df):    
    # do iterative imputing for all missing values
    imp = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=50), max_iter=500, verbose=True)
    imputed = pd.DataFrame(imp.fit_transform(df))
    imputed.columns = df.columns
    imputed.index = df.index
    
    return imputed


def direction_binning(dirs, bin_degree=45):

    N = 360//bin_degree
    # ticks = np.r_[0:360:bin_degree]
    edge = np.r_[0,bin_degree/2:360:bin_degree, 360]    
    # lab4 = ['N', 'E', 'S', 'W', 'NN']
    # lab8 = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'NN']
    # lab16 = ['N', 'NNE', 'NE', 'NEE', 'E', 'SEE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'SWW', 'W', 'NWW', 'NW', 'NNW','NN']
    c = pd.cut(dirs, edge, right=True, labels=np.arange(0,N+1))
   
    #combine the last group with the first group (angles range around zero)
    
    return c.replace(N, 0).cat.remove_unused_categories() 


# plot wind direciton
def plot_wind(directions:pd.Series, bin_degree=45, ax=None):

    if ax is None:
        ax = plt.subplot(111,polar=True)
    
    N = 360//bin_degree
    d = direction_binning(pd.to_numeric(directions, errors='coerce'), bin_degree=bin_degree)

    # create theta for N directions
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    radii = d.value_counts(sort=False).reindex(np.arange(0,N, dtype='int'), fill_value=0).values
    # width of each bin on the plot
    width = (2*np.pi) / N 

    # make a polar plot
#     plt.figure(figsize = (12, 8))
    bars = ax.bar(theta, radii, width=width, bottom=0, align='center',
                  edgecolor = 'gray', linewidth=2)

    # set the lable go clockwise and start from the top
    ax.set_theta_zero_location("N")
    # clockwise
    ax.set_theta_direction(-1)

    # set the label
    # ax.set_xticklabels(ticks)
    ax.set_rlabel_position(140)

    #plt.show()
    return bars


def plot_wind_gust(directions:pd.Series, directionsG:pd.Series, G_dir, G_speed, bin_degree=45, ax=None):

    if ax is None:
        ax = plt.subplot(111,polar=True)
    
    N = 360//bin_degree
    d = direction_binning(pd.to_numeric(directions, errors='coerce'), bin_degree=bin_degree)
    dG = direction_binning(pd.to_numeric(directionsG, errors='coerce'), bin_degree=bin_degree)
    # create theta for N directions
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    radii = d.value_counts(sort=False).reindex(np.arange(0,N, dtype='int'), fill_value=0).values
    radiiG = dG.value_counts(sort=False).reindex(np.arange(0,N, dtype='int'), fill_value=0).values
    # width of each bin on the plot
    width = (2*np.pi) / N 

    # make a polar plot
    bars = ax.bar(theta, radii, width=width, bottom=0, align='center',
                  edgecolor = 'gray', linewidth=2, color="#1f77b4")
    barsG = ax.bar(theta, radiiG, width=width, bottom=0, align='center',
                  edgecolor = 'gray', linewidth=2, alpha=0.7, color="#ff7f0e")
    # plot the most frequent wind gust direction
    radi_max = max(radii.max(), radiiG.max())*1
#     line = ax.vlines(np.radians(G_dir),0,radi_max, color='r')
    ax.quiver(0,0,np.radians(G_dir),radi_max, color='r', angles="xy", 
              units="y", scale=1, scale_units='xy',headwidth=5,
              zorder=10)
#     ax.annotate(f'max wind gust:\n{G_speed}m/s',
#             xy=(np.radians(G_dir), radi_max),  # theta, radius
#             xytext=(np.radians(G_dir), radi_max), 
#             fontsize=14
#             )
    ax.text(1.0, 1.0, f'max gust:\n{G_speed}m/s',
        ha='right', va='top',color='r',
        transform=ax.transAxes)
    ax.text(0, 1.0, 'Wind Dir.',
        ha='left', va='top',color='#1f77b4',
        transform=ax.transAxes)
    ax.text(0, 0.95, 'Gust Dir.',
        ha='left', va='top',color='#ff7f0e',
        transform=ax.transAxes)
    # set the lable go clockwise and start from the top
    ax.set_theta_zero_location("N")
    # clockwise
    ax.set_theta_direction(-1)

    # remove grid and polar circular axis frame, y ticks labels
    ax.grid(alpha=0.5)
    ax.spines['polar'].set_visible(False)
    ax.set_yticks([])
    # set the angle label
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels(['N', '', 'E', '', 'S', '', 'W', ''])
#     ax.set_rlabel_position(140)
    
    
    #plt.show()
    return radi_max


def plot_wind_color(df_temp, col_bin, col_ws, ws_max=None, G_dir=None, G_speed=None, cmap='jet', ax=None):
    '''rose plot for wind direciton.
    Polar histogram height(size) stands for occurance(count) in that direction.
    Color of the bar stands for median wind speed in that direction.
    '''

    radii = df_temp[col_bin].value_counts(sort=False)
    ws = df_temp.groupby([col_bin])[col_ws].median().fillna(0)
    # create cmap
    cmap = matplotlib.cm.get_cmap(cmap)
    #scale ws to [0,1], use either given global max, or local max
    if ws_max is None:
        ws_max = ws.max()
    ws_scaled = (ws-ws.min())/(ws_max-ws.min())
    
    #get corresponding colors
    colors = [cmap(w) for w in ws_scaled]
    #bar width
    width = (2*np.pi) / 36
    #polar coord of the bar (starting angles in radian)
    theta = np.linspace(0.0, 2 * np.pi, 36, endpoint=True)
    radi_max = radii.max()
    
    if ax is None:
        ax = plt.subplot(111,polar=True)

    bars = ax.bar(theta, radii, width=width, bottom=0, align='edge',
                  edgecolor = 'gray', linewidth=0, color=colors)
    
    if (G_dir is not None) & (G_speed is not None):
        ax.quiver(0,0,np.radians(G_dir),radi_max, color='r', angles="xy", 
              units="y", scale=1, scale_units='xy',headwidth=5,
              zorder=10)
        ax.text(1.0, 1.0, f'max gust:\n{G_speed}m/s',
            ha='right', va='top',color='r', fontsize=12,
            transform=ax.transAxes)
    
    # set starting from 12o'clock and clockwise
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    # hide grid, spine and yticks
    ax.grid(alpha=0.3)
    ax.spines['polar'].set_visible(False)
    ax.set_yticks([])
    # set the angle label
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
#     ax.set_xticks([0,np.pi/2,np.pi,np.pi*3/2])
#     ax.set_xticklabels(['N','', 'E','', 'S','', 'W',''])
    ax.set_xticklabels(['','', '','', '','', '',''])


#     plt.show()
    return bars


def map_with_marker(lat: pd.Series,
                    lng: pd.Series,
                    label1: pd.Series,
                    label2: pd.Series,
                    color: pd.Series,
                    cmap='Set1',
                    location='Taiwan',
                    width=600,
                    height=1000,
                    zoom=8,
                    **kwarg):
    
    # create a color map
    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap, color.nunique())
        rgb = [matplotlib.colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
    else:
        rgb=cmap

    # create the location map
    g = geocoder.osm(location)
    latitude = g.latlng[0]
    longitude = g.latlng[1]
    
    f = folium.Figure(width=width, height=height)
    map_ = folium.Map(location=[latitude, longitude], zoom_start=zoom).add_to(f)
    folium.raster_layers.TileLayer(tiles='stamenterrain', name='地型圖').add_to(map_)
    folium.LayerControl(collapsed=True, position='topleft').add_to(map_)

    # add marker to map
    for lat, lng, l1, l2, c in zip(lat, lng, label1, label2, color):
        text = f'{l1}, {l2}'
        label = folium.Popup(text, parse_html=True, max_width=200)
        folium.CircleMarker(
            [lat, lng],
            radius=5,
            popup=label,
            tooltip=text,
            color=rgb[c],
            fill=True,
            fill_color=rgb[c],
            fill_opacity=0.9,
            parse_html=False).add_to(map_)
    
    return map_
