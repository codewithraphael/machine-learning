# Bike Sharing Dataset

## Introduction

Bike sharing systems are new generation of traditional bike rentals where the entire process from membership, rental, and return back has become automatic. These systems, with their high levels of automation, offer a new perspective on transportation, both in terms of traffic efficiency, environmental impact, and health outcomes. 

In the current context, there are more than 500 bike-sharing programs worldwide, with over 500, 000 bicycles. The system's ability to collect detailed data on both hourly and daily basis and integrate weather and seasonal information makes it a valuable tool for monitoring and analyzing mobility patterns in cities.

## Associated Tasks

1. **Regression**: Predict the bike rental count hourly or daily based on the environmental and seasonal settings.
2. **Event and Anomaly Detection**: Count of rented bikes are correlated to various events in the town, such as hurricanes and other significant weather phenomena. These events can be identified using search engine queries and validated using anomaly detection algorithms.

## Dataset Characteristics

The dataset contains aggregated data for both hourly and daily rental counts for the Capital Bikeshare system in Washington D.C., USA. Each record includes the following fields:

- `instant`: The record index.
- `dteday`: The date of the rental.
- `season`: The season of the year, categorized as 1 for spring, 2 for summer, 3 for fall, and 4 for winter.
- `yr`: The year of the rental (0 for 2011, 1 for 2012).
- `mnth`: The month of the rental (1 to 12).
- `hr`: The hour of the rental.
- `holiday`: Indicates if the rental day is a holiday or not.
- `weekday`: The day of the week.
- `workingday`: A binary indicator, 1 if the day is neither a weekend nor a holiday, otherwise 0.
- `weathersit`: The weather condition, categorized as 1 for clear skies, 2 for light rain or scattered clouds, 3 for heavy rain with ice, and 4 for snow or fog.
- `temp`: The normalized temperature in Celsius, divided by 41 (maximum).
- `atemp`: The normalized feeling temperature in Celsius, divided by 50 (maximum).
- `hum`: The normalized humidity, divided by 100 (maximum).
- `windspeed`: The normalized wind speed, divided by 67 (maximum).
- `casual`: The number of casual users.
- `registered`: The number of registered users.
- `cnt`: The total number of rentals including both casual and registered.

## License

Use of this dataset in publications must be cited to the following publication:

Fanaee-T, Hadi, and Gama, Joao, "Event labeling combining ensemble detectors and background knowledge", Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg, doi:10.1007/s13748-013-0040-3.

@article{
	year={2013},
	issn={2192-6352},
	journal={Progress in Artificial Intelligence},
	doi={10.1007/s13748-013-0040-3},
	title={Event labeling combining ensemble detectors and background knowledge},
	url={http://dx.doi.org/10.1007/s13748-013-0040-3},
	publisher={Springer Berlin Heidelberg},
	keywords={Event labeling; Event detection; Ensemble learning; Background knowledge},
	author={Fanaee-T, Hadi and Gama, Joao},
	pages={1-15}
}

## Contact

For further information about this dataset please contact Hadi Fanaee-T (hadi.fanaee@fe.up.pt).