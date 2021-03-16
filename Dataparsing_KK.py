import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Q1
def parse_cardata():
    cardata = pd.read_csv('cardata.csv')
    return cardata
cardata = parse_cardata()
cardata.head()
print(type(cardata))


# Q2
cardata['age'].head()
print(cardata['age'].min())
print(cardata['age'].max())

bucket_size = 5
age_keys = []
for i in range(cardata['age'].min(), cardata['age'].max(), bucket_size):
    age_keys.append(str(i)+'-'+str(i+bucket_size))
age_keys

cars_keys = []
for car in cardata['car']:
    if car not in cars_keys:
        cars_keys.append(car)
cars_keys
cardata.drop_duplicates(subset=['car'], keep='last')
print(cars_keys)
print(age_keys)

def cars_vs_age(cardata,bucketsize=5):
    age_keys = []
    for i in range(cardata['age'].min(), cardata['age'].max(), bucketsize):
        age_keys.append(str(i)+'-'+str(i+bucketsize))
    cars_keys = []
    for car in cardata['car']:
        if car not in cars_keys:
            cars_keys.append(car)
    histogram ={}
    for car in cars_keys:
        if car not in histogram:
            histogram[car] = {}
        for ag1 in age_keys:
            histogram[car][ag1] = 0
            ag = ag1.split('-')
            min_age = int(ag[0])
            max_age = int(ag[1])
            histogram[car][ag1] = int(np.sum((cardata['age']>= min_age) & (cardata['age']<max_age) & (cardata['car'] == car)))
                
    return histogram
histogram = cars_vs_age(cardata,5)
plt.bar(histogram['i20'].keys(), histogram['i20'].values())

# Q3
def age_vs_cars(cardata,bucketsize=5):
    age_keys = []
    for i in range(cardata['age'].min(), cardata['age'].max(), bucketsize):
        age_keys.append(str(i)+'-'+str(i+bucketsize))
    cars_keys = []
    for car in cardata['car']:
        if car not in cars_keys:
            cars_keys.append(car)
    histogram ={}
    for ag1 in age_keys:
        if ag1 not in histogram:
            histogram[ag1] = {}
        for car in cars_keys:
            histogram[ag1][car] = 0
            ag = ag1.split('-')
            min_age = int(ag[0])
            max_age = int(ag[1])
            histogram[ag1][car] = int(np.sum((cardata['age']>= min_age) & (cardata['age']<max_age) & (cardata['car'] == car)))
    print(histogram)            
    return histogram
print(histogram)



# Q4
bucketsize=5
age_keys = []
for i in range(cardata['age'].min(), cardata['age'].max(), bucketsize):
    age_keys.append(str(i)+'-'+str(i+bucketsize))

cars_keys = []
for car in cardata['car']:
    if car not in cars_keys:
        cars_keys.append(car)
new_hist = {}
for i in age_keys:
    new_hist[i] = {}
    for car in cars_keys:
        new_hist[i][car] = histogram[car][i]
print(new_hist)


def company_vs_cars(cardata):
    company_vs_cars_map ={}
    car_names = []
    car_n = []
    uniq_comp= cardata['company'].unique()
    for company in uniq_comp:
        company_vs_cars_map[company] = []
        uniq_cars = cardata['car'].unique()
        for car in uniq_cars:
            car_no = int(np.sum((cardata['car'] == car) & (cardata['company'] == company)))
            company_vs_cars_map[company].append((car, car_no))
        company_vs_cars_map[company].sort(key = lambda x: x[1], reverse=True)
            
    print(company_vs_cars_map)
    return company_vs_cars_map
company_vs_cars_map = company_vs_cars(cardata)

# Q5
car_points = {}
for car in cardata['car'].unique():
    car_points[car] = 0
    for company in cardata['company'].unique():
        print(company_vs_cars_map[company][i][0] for i in range(2))
print(car_points)