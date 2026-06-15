from goseidataset import TimeSeriesGenerator

data = {
    "value": [1, 2, 3, 4, 5]
}

ts = TimeSeriesGenerator(data)

print("\nYEAR")
print(ts.year_series(start="2020"))

print("\nMONTH")
print(ts.month_series(start="2020-01-01"))

print("\nWEEK")
print(ts.week_series(start="2020-01-01"))

print("\nDAY")
print(ts.day_series(start="2020-01-01"))

print("\nHOUR")
print(ts.hour_series(start="2020-01-01 00:00:00"))

print("\nMINUTE")
print(ts.minute_series(start="2020-01-01 00:00:00"))

print("\nSECOND")
print(ts.second_series(start="2020-01-01 00:00:00"))

print("\nTIMESTAMP")
print(ts.timestamp_series(start="2020-01-01 00:00:00"))