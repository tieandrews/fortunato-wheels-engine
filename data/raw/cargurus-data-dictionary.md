The original data source come from this Kaggle dataset: https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset/code?datasetId=885427&sortBy=voteCount

## Data Description

1. vin: Type String. Vehicle Identification Number is a unique encoded string for every vehicle. Read more at https://www.autocheck.com/vehiclehistory/vin-basics
2. back_legroom: Type String. Legroom in the rear seat.
3. bed: Type String. Category of bed size(open cargo area) in pickup truck. Null usually means the vehicle isn't a pickup truck
4. bed_height: Type String. Height of bed in inches
5. bed_length: Type String. Length of bed in inches
6. body_type: Type String. Body Type of the vehicle. Like Convertible, Hatchback, Sedan, etc.
7. cabin: Type String. Category of cabin size(open cargo area) in pickup truck. Eg: Crew Cab, Extended Cab, etc.
8. city: Type String. city where the car is listed. Eg: Houston, San Antonio, etc.
9. city_fuel_economy: Type Float. Fuel economy in city traffic in km per litre
10. combine_fuel_economy: Type Float. Combined fuel economy is a weighted average of City and Highway fuel economy in km per litre
11. daysonmarket: Type Integer. Days since the vehicle was first listed on the website.
12. dealer_zip: Type Integer. Zipcode of the dealer
13. description: Type String. Vehicle description on the vehicle's listing page
14. engine_cylinders: Type String. The engine configuration. Eg: I4, V6, etc.
15. engine_displacement: Type Float. engine_displacement is the measure of the cylinder volume swept by all of the pistons of a piston engine, excluding the combustion chambers.
16. engine_type: Type String. The engine configuration. Eg: I4, V6, etc.
17. exterior_color: Type String. Exterior color of the vehicle, usually a fancy one same as the brochure.
18. fleet: Type Boolean. Whether the vehicle was previously part of a fleet.
19. frame_damaged: Type Boolean. Whether the vehicle has a damaged frame.
20. franchise_dealer: Type Boolean. Whether the dealer is a franchise dealer.
21. franchise_make: Type String. The company that owns the franchise.
22. front_legroom: Type String. The legroom in inches for the passenger seat
23. fuel_tank_volume: Type String. Fuel tank's filling capacity in gallons
24. fuel_type: Type String. Dominant type of fuel ingested by the vehicle.
25. has_accidents: Type Boolean. Whether the vin has any accidents registered.
26. height: Type String. Height of the vehicle in inches
27. highway_fuel_economy: Type Float. Fuel economy in highway traffic in km per litre
28. horsepower: Type Float. Horsepower is the power produced by an engine.
29. interior_color: Type String. Interior color of the vehicle, usually a fancy one same as the brochure.
30. isCab: Type Boolean. Whether the vehicle was previously taxi/cab.
31. is_certified: Type Boolean. Whether the vehicle is certified. Certified cars are covered through warranty period
32. is_cpo: Type Boolean. Pre-owned cars certified by the dealer. Certified vehicles come with a manufacturer warranty for free repairs for a certain time period. Read more at https://www.cartrade.com/blog/2015/auto-guides/pros-and-cons-of-buying-a-certified-pre-owned-car-1235.html
33. is_new: Type Boolean. If True means the vehicle was launched less than 2 years ago.
34. is_oemcpo: Type Boolean. Pre-owned cars certified by the manufacturer. Read more at https://www.cargurus.com/Cars/articles/know_the_difference_dealership_cpo_vs_manufacturer_cpo
35. latitude: Type Float. Latitude from the geolocation of the dealership.
36. length: Type String. Length of the vehicle in inches
37. listed_date: Type String. The date the vehicle was listed on the website. Does not make days_on_market obsolete. The prices is days_on_market days after the listed date.
38. listing_color: Type String. Dominant color group from the exterior color.
39. listing_id: Unique Type Integer. Listing id from the website
40. longitude: Type Float. Longitude from the geolocation of the dealership.
41. main_picture_url: Type String.
42. major_options: Type String. Major options of the vehicle in list format.
43. make_name: Type String. Make name of the vehicle.
44. maximum_Seating: Type Integer. Maximum seating capacity of the vehicle.
45. mileage: Type Integer. Mileage of the vehicle in miles.
46. model_name: Type String. Model name of the vehicle.
47. owner_count: Type Integer. Number of owners the vehicle has had.
48. power: Type String. Power of the vehicle in hp.
49. price: Type Integer. Price of the vehicle in USD.
50. salvage: Type Boolean. Whether the vehicle is salvaged.
51. savings_amount: Type Integer. The amount of savings the dealer is offering on the vehicle.
52. seller_rating: Type Float. The rating of the dealer on the website.
53. sp_id: Type Integer. Unique id of the seller.
54. sp_name: Type String. Name of the seller.
55. theft_title: Type Boolean. Whether the vehicle has a theft title.
56. torque: Type String. Torque of the vehicle in lb-ft.
57. transmission: Type String. Transmission type of the vehicle.
58. transmission_display: Type String. Transmission type of the vehicle.
59. trim_id: Type Integer. Trim id of the vehicle.
60. vehicle_damage_category: Type String. Category of damage the vehicle has.
61. wheel_system: Type String. Wheel system of the vehicle, FWD, AWD etc. 
62. wheelbase: Type String. Wheelbase of the vehicle in inches.
63. year: Type Integer. Year of the vehicle.