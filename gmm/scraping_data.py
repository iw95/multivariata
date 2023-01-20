import csv
import json
import codecs
import urllib.request
import urllib.error
import sys
import numpy as np

def download_data():
    #  Code from https://www.visualcrossing.com/resources/blog/how-to-load-historical-weather-data-using-python-without-scraping/

    BaseURL = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/'

    ApiKey='NX4H8P2664Q8N2ZZRNTCEEAA6'
    # UnitGroup sets the units of the output - us or metric
    UnitGroup='metric'

    # Location for the weather data
    Location='Florence,Italy'

    # Start and end dates
    StartDate = '2022-01-01'
    EndDate='2022-12-31'

    # JSON or CSV
    # JSON format supports daily, hourly, current conditions, weather alerts and events in a single JSON package
    # CSV format requires an 'include' parameter below to indicate which table section is required
    ContentType="csv"

    # include sections (days,hours,current,alerts)
    Include="days"

    # basic query including location, start and end date
    ApiQuery=BaseURL + Location + "/"+StartDate + "/"+EndDate
    # append parameters
    ApiQuery += "?unitGroup="+UnitGroup + "&contentType="+ContentType + "&include="+Include
    # api key
    ApiQuery+="&key="+ApiKey


    print(' - Running query URL: ', ApiQuery)

    try:
        CSVBytes = urllib.request.urlopen(ApiQuery)
    except urllib.error.HTTPError as e:
        ErrorInfo = e.read().decode()
        print('Error code: ', e.code, ErrorInfo)
        sys.exit()
    except urllib.error.URLError as e:
        ErrorInfo = e.read().decode()
        print('Error code: ', e.code, ErrorInfo)
        sys.exit()

    # Parse the results as CSV
    CSVText = csv.reader(codecs.iterdecode(CSVBytes, 'utf-8'))
    with open('data/full_data.csv', 'w') as f_full, open('data/selected_data.csv', 'w') as f_sel:
        # create the csv writer
        writer_full = csv.writer(f_full)
        writer_sel = csv.writer(f_sel)
        first = True
        for row in CSVText:
            if first:
                labels = row
                int_lab_old = ['datetime','tempmax', 'tempmin', 'humidity', 'precip', 'precipcover', 'cloudcover',
                           'solarradiation']  # 'windspeed','sealevelpressure','visibility'
                int_lab = ['datetime', 'temp', 'windspeed', 'precip', 'cloudcover', 'visibility', 'sealevelpressure',
                               'solarradiation']  # 'sealevelpressure'
                idxs = [labels.index(lab) for lab in int_lab]
                first = False
            # write a row to the csv file
            writer_full.writerow(row)
            writer_sel.writerow(np.array(row)[idxs])


def newly_select():
    with open('data/full_data.csv', 'r') as f_full, open('data/selected_data.csv', 'w') as f_sel:
        # create the csv writer and reader
        reader = csv.reader(f_full)
        writer_sel = csv.writer(f_sel)
        first = True
        for row in reader:
            if first:
                labels = row
                int_lab_old = ['datetime','tempmax', 'tempmin', 'humidity', 'precip', 'precipcover', 'cloudcover',
                           'solarradiation']  # 'windspeed','sealevelpressure','visibility'
                int_lab = ['datetime', 'temp', 'windspeed', 'precip', 'cloudcover', 'visibility', 'sealevelpressure',
                               'solarradiation']  # 'sealevelpressure'
                idxs = [labels.index(lab) for lab in int_lab]
                first = False
            # write a row to the csv file
            writer_sel.writerow(np.array(row)[idxs])


def main():
    newly_select()


if __name__ == '__main__':
    main()