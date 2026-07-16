TARGET_COLUMN = "claim_status"

DROP_COLUMNS = [
    "policy_id"
]

NUMERIC_FEATURES = [
    "subscription_length",
    "vehicle_age",
    "customer_age",
    "region_density",
    "airbags",
    "displacement",
    "cylinder",
    "turning_radius",
    "length",
    "width",
    "gross_weight",
    "ncap_rating"
]

CATEGORICAL_FEATURES = [
    "region_code",
    "segment",
    "model",
    "fuel_type",
    "max_torque",
    "max_power",
    "engine_type",
    "is_esc",
    "is_adjustable_steering",
    "is_tpms",
    "is_parking_sensors",
    "is_parking_camera",
    "rear_brakes_type",
    "transmission_type",
    "steering_type",
    "is_front_fog_lights",
    "is_rear_window_wiper",
    "is_rear_window_washer",
    "is_rear_window_defogger",
    "is_brake_assist",
    "is_power_door_locks",
    "is_central_locking",
    "is_power_steering",
    "is_driver_seat_height_adjustable",
    "is_day_night_rear_view_mirror",
    "is_ecw",
    "is_speed_alert"
]