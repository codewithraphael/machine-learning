# Insurance Claims Dataset

## Overview
This dataset contains information about vehicle insurance policies and their claim status. It includes customer demographics, vehicle specifications, safety features, and whether a claim was filed. The dataset is valuable for building predictive models to identify high-risk policies and understand factors that influence insurance claims.

## Dataset Information
- **File**: `Insurance claims data.csv`
- **Total Features**: 39
- **Target Variable**: `claim_status` (binary: 0 = No claim, 1 = Claim filed)

## Features Description

### Policy Information
- **policy_id**: Unique identifier for each insurance policy
- **subscription_length**: Length of the insurance subscription (in years)
- **claim_status**: Target variable indicating whether a claim was filed (0/1)

### Customer Demographics
- **customer_age**: Age of the customer (in years)
- **region_code**: Geographic region code (e.g., C1, C2, C3, etc.)
- **region_density**: Population density of the region
- **segment**: Customer segment classification (A, B1, B2, C1, C2, Utility)

### Vehicle Characteristics
- **model**: Vehicle model identifier (M1, M2, M3, etc.)
- **vehicle_age**: Age of the vehicle (in years)
- **fuel_type**: Type of fuel (Diesel, Petrol, CNG)
- **engine_type**: Engine specification (e.g., "1.5 L U2 CRDi", "K Series Engine")
- **displacement**: Engine displacement in cubic centimeters (cc)
- **cylinder**: Number of cylinders in the engine
- **max_torque**: Maximum torque specification (e.g., "250Nm@2750rpm")
- **max_power**: Maximum power specification (e.g., "113.45bhp@4000rpm")

### Transmission & Steering
- **transmission_type**: Type of transmission (Manual, Automatic)
- **steering_type**: Type of steering (Manual, Power, Electric)
- **turning_radius**: Turning radius of the vehicle (in meters)

### Vehicle Dimensions & Weight
- **length**: Length of the vehicle (in mm)
- **width**: Width of the vehicle (in mm)
- **gross_weight**: Gross vehicle weight (in kg)

### Safety Features (Boolean)
- **airbags**: Number of airbags in the vehicle
- **is_esc**: Electronic Stability Control
- **is_adjustable_steering**: Adjustable steering wheel
- **is_tpms**: Tire Pressure Monitoring System
- **is_parking_sensors**: Parking sensor availability
- **is_parking_camera**: Parking camera availability
- **is_front_fog_lights**: Front fog lights
- **is_rear_window_wiper**: Rear window wiper
- **is_rear_window_washer**: Rear window washer
- **is_rear_window_defogger**: Rear window defogger
- **is_brake_assist**: Brake assist system
- **is_power_door_locks**: Power door locks
- **is_central_locking**: Central locking system
- **is_power_steering**: Power steering
- **is_driver_seat_height_adjustable**: Driver seat height adjustment
- **is_day_night_rear_view_mirror**: Day/night rear-view mirror
- **is_ecw**: Electronic Control Warning
- **is_speed_alert**: Speed alert system

### Brake System
- **rear_brakes_type**: Type of rear brakes (Disc, Drum)

### Vehicle Safety Rating
- **ncap_rating**: NCAP safety rating (0-5 scale)

## Use Cases
1. **Claim Prediction**: Build classification models to predict the likelihood of insurance claims
2. **Risk Assessment**: Identify high-risk vehicle and customer combinations
3. **Feature Analysis**: Understand which vehicle features and customer attributes impact claim rates
4. **Pricing Optimization**: Use insights to optimize insurance premium pricing
5. **Customer Segmentation**: Segment customers based on claim likelihood

## Data Exploration Ideas
- Distribution of claims across different vehicle fuel types
- Impact of vehicle safety features on claim rates
- Relationship between customer age and claims
- Regional patterns in insurance claims
- Effect of vehicle age on claim likelihood
- Correlation between transmission type and claims

## Potential Challenges
- **Imbalanced Classes**: The dataset may have imbalanced claim distribution (0 vs 1)
- **Feature Engineering**: Vehicle specifications contain mixed data types (strings with specifications)
- **Categorical Variables**: Multiple categorical features that require encoding
- **Multicollinearity**: Some vehicle specifications may be correlated

## Recommended Analysis
1. **Exploratory Data Analysis (EDA)**: Visualize distributions and relationships
2. **Data Preprocessing**: Handle missing values, encode categorical variables
3. **Feature Selection**: Identify most important features for prediction
4. **Model Building**: Test multiple classifiers (Logistic Regression, Random Forest, XGBoost, etc.)
5. **Model Evaluation**: Use appropriate metrics (precision, recall, F1-score, AUC-ROC)

## References
This dataset can be used for insurance analytics, predictive modeling, and risk assessment studies in the automotive insurance industry.
