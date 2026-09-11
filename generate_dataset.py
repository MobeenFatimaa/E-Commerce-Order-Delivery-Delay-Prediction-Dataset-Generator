import numpy as np
import polars as pl
from datetime import date, timedelta

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
NUM_RECORDS = 100_000
SEED = 42
OUTPUT_FILE = "ecommerce_delivery_predictions.csv"

np.random.seed(SEED)

# -----------------------------------------------------------------------------
# Category & Geographic Setup
# -----------------------------------------------------------------------------
CATEGORIES = [
    "Electronics", "Clothing", "Home & Kitchen", "Books", 
    "Beauty", "Sports", "Toys", "Automotive"
]
PAYMENT_METHODS = ["Credit Card", "Debit Card", "PayPal", "UPI", "Cash on Delivery"]
ORDER_PRIORITIES = ["Low", "Medium", "High", "Critical"]
SHIPPING_METHODS = ["Standard", "Express", "Same-Day", "Economy"]
CARRIERS = ["FedEx", "UPS", "DHL", "USPS", "OnTrac"]
WEATHER_CONDITIONS = ["Clear", "Rain", "Snow", "Fog", "Storm"]
TRAFFIC_LEVELS = ["Low", "Medium", "High", "Jam"]

CITIES_STATES = [
    ("New York", "NY"), ("Los Angeles", "CA"), ("Chicago", "IL"),
    ("Houston", "TX"), ("Phoenix", "AZ"), ("Philadelphia", "PA"),
    ("San Antonio", "TX"), ("San Diego", "CA"), ("Dallas", "TX")
]

# -----------------------------------------------------------------------------
# Generating Core Features
# -----------------------------------------------------------------------------
# Customer (7 features)
customer_ids_str = [f"CUST-{cid}" for cid in np.random.randint(10000, 99999, size=NUM_RECORDS)]
customer_ages = np.random.randint(18, 70, size=NUM_RECORDS)
customer_genders = np.random.choice(["Male", "Female", "Other"], size=NUM_RECORDS, p=[0.48, 0.48, 0.04])

city_idx = np.random.choice(len(CITIES_STATES), size=NUM_RECORDS)
customer_cities = [CITIES_STATES[i][0] for i in city_idx]
customer_states = [CITIES_STATES[i][1] for i in city_idx]

customer_incomes = np.random.lognormal(mean=10.8, sigma=0.5, size=NUM_RECORDS).round(2)
customer_ratings = np.random.uniform(1.0, 5.0, size=NUM_RECORDS).round(1)

# Order (9 features)
order_ids_str = [f"ORD-{i+100000:06d}" for i in range(NUM_RECORDS)]
start_date = date(2025, 1, 1)
random_days = np.random.randint(0, 365, size=NUM_RECORDS)
order_dates = [start_date + timedelta(days=int(d)) for d in random_days]

product_categories = np.random.choice(CATEGORIES, size=NUM_RECORDS)
product_prices = np.random.exponential(scale=50.0, size=NUM_RECORDS).clip(5, 1500).round(2)
quantities = np.random.poisson(lam=1.5, size=NUM_RECORDS).clip(1, 10)
discount_percents = np.random.choice([0, 5, 10, 15, 20, 25, 30], size=NUM_RECORDS, p=[0.4, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05])
order_values = ((product_prices * quantities) * (1 - discount_percents / 100.0)).round(2)
payment_methods = np.random.choice(PAYMENT_METHODS, size=NUM_RECORDS)
order_priorities = np.random.choice(ORDER_PRIORITIES, size=NUM_RECORDS, p=[0.2, 0.5, 0.2, 0.1])

# Seller (5 features)
seller_ids_str = [f"SELL-{np.random.randint(1000, 9999)}" for _ in range(NUM_RECORDS)]
seller_ratings = np.random.uniform(2.0, 5.0, size=NUM_RECORDS).round(1)
seller_experience_years = np.random.randint(1, 15, size=NUM_RECORDS)
s_city_idx = np.random.choice(len(CITIES_STATES), size=NUM_RECORDS)
seller_cities = [CITIES_STATES[i][0] for i in s_city_idx]
seller_reliability_scores = (seller_ratings * 18 + seller_experience_years * 1.2 + np.random.normal(0, 3, NUM_RECORDS)).clip(10, 100).round(1)

# Shipping (9 features)
warehouse_distances = np.random.gamma(shape=2.0, scale=250.0, size=NUM_RECORDS).clip(10, 3000).round(1)
shipping_methods = np.random.choice(SHIPPING_METHODS, size=NUM_RECORDS, p=[0.5, 0.3, 0.1, 0.1])
carriers = np.random.choice(CARRIERS, size=NUM_RECORDS)
package_weights = np.random.exponential(scale=3.0, size=NUM_RECORDS).clip(0.1, 50.0).round(2)
package_volumes = (package_weights * np.random.uniform(200, 600, size=NUM_RECORDS)).round(1)
processing_time_hours = np.random.gamma(shape=1.5, scale=12.0, size=NUM_RECORDS).clip(2, 96).round(1)
dispatch_delay_hours = np.random.exponential(scale=4.0, size=NUM_RECORDS).clip(0, 72).round(1)

base_days_map = {"Same-Day": 1, "Express": 2, "Standard": 5, "Economy": 7}
est_days = np.array([base_days_map[m] for m in shipping_methods]) + (warehouse_distances / 500).astype(int)
estimated_delivery_days = est_days.clip(1, 14)

# External Factors (5 features)
weather_conditions = np.random.choice(WEATHER_CONDITIONS, size=NUM_RECORDS, p=[0.6, 0.2, 0.08, 0.07, 0.05])
traffic_levels = np.random.choice(TRAFFIC_LEVELS, size=NUM_RECORDS, p=[0.4, 0.35, 0.2, 0.05])
month = np.array([d.month for d in order_dates])
day_of_week = np.array([d.weekday() for d in order_dates])
holiday_period = np.where(np.isin(month, [11, 12]), 1, 0)
weekend_order = np.where(day_of_week >= 5, 1, 0)
peak_season = np.where(np.isin(month, [10, 11, 12]), 1, 0)

# Derived Features (5 features)
distance_to_delivery_ratio = (warehouse_distances / estimated_delivery_days).round(2)
shipping_risk_score = (
    (warehouse_distances / 300) * 1.5 + 
    (package_weights / 5) * 1.2 + 
    (dispatch_delay_hours / 6) * 2.0 +
    np.where(weather_conditions == "Storm", 15, np.where(weather_conditions == "Snow", 10, 0)) +
    np.where(traffic_levels == "Jam", 12, np.where(traffic_levels == "High", 6, 0))
).clip(0, 100).round(1)
seller_performance_score = (seller_reliability_scores * 0.7 + seller_ratings * 6).clip(0, 100).round(1)
customer_order_frequency = np.random.poisson(lam=4, size=NUM_RECORDS).clip(1, 30)
delivery_efficiency_score = (100 - (processing_time_hours / 96 * 50) - (dispatch_delay_hours / 72 * 50)).clip(0, 100).round(1)

# -----------------------------------------------------------------------------
# Target Generation (1 Target)
# -----------------------------------------------------------------------------
# Base log-odds tuned to achieve ~25% positive late_delivery rate
log_odds = -1.85 

log_odds += (dispatch_delay_hours / 12.0) * 0.8
log_odds += (processing_time_hours / 24.0) * 0.3
log_odds += (warehouse_distances / 1000.0) * 0.4
log_odds += (package_weights / 10.0) * 0.25
log_odds -= (seller_reliability_scores / 100.0) * 0.8
log_odds += np.where(weather_conditions == "Storm", 1.2, np.where(weather_conditions == "Snow", 0.7, 0.0))
log_odds += np.where(traffic_levels == "Jam", 0.9, np.where(traffic_levels == "High", 0.4, 0.0))
log_odds += np.where(shipping_methods == "Same-Day", 0.6, np.where(shipping_methods == "Express", -0.3, 0.0))
log_odds += holiday_period * 0.4 + peak_season * 0.3

late_prob = 1.0 / (1.0 + np.exp(-log_odds))
late_delivery = (np.random.uniform(0, 1, size=NUM_RECORDS) < late_prob).astype(int)

actual_delivery_days = np.where(
    late_delivery == 1,
    estimated_delivery_days + np.random.randint(1, 6, size=NUM_RECORDS),
    np.maximum(1, estimated_delivery_days - np.random.randint(0, 2, size=NUM_RECORDS))
)

# -----------------------------------------------------------------------------
# Construct DataFrame (Exactly 40 Columns)
# -----------------------------------------------------------------------------
data_dict = {
    # Customer (7)
    "customer_id": customer_ids_str,
    "customer_age": customer_ages,
    "customer_gender": customer_genders,
    "customer_city": customer_cities,
    "customer_state": customer_states,
    "customer_income": customer_incomes,
    "customer_rating": customer_ratings,
    # Order (9)
    "order_id": order_ids_str,
    "order_date": order_dates,
    "product_category": product_categories,
    "product_price": product_prices,
    "quantity": quantities,
    "discount_percent": discount_percents,
    "order_value": order_values,
    "payment_method": payment_methods,
    "order_priority": order_priorities,
    # Seller (5)
    "seller_id": seller_ids_str,
    "seller_rating": seller_ratings,
    "seller_experience_years": seller_experience_years,
    "seller_city": seller_cities,
    "seller_reliability_score": seller_reliability_scores,
    # Shipping (8 - excluding actual_delivery_days from base predictors)
    "warehouse_distance_km": warehouse_distances,
    "shipping_method": shipping_methods,
    "carrier": carriers,
    "package_weight_kg": package_weights,
    "package_volume": package_volumes,
    "processing_time_hours": processing_time_hours,
    "dispatch_delay_hours": dispatch_delay_hours,
    "estimated_delivery_days": estimated_delivery_days,
    # External Factors (5)
    "weather_condition": weather_conditions,
    "traffic_level": traffic_levels,
    "holiday_period": holiday_period,
    "weekend_order": weekend_order,
    "peak_season": peak_season,
    # Derived Features (5)
    "distance_to_delivery_ratio": distance_to_delivery_ratio,
    "shipping_risk_score": shipping_risk_score,
    "seller_performance_score": seller_performance_score,
    "customer_order_frequency": customer_order_frequency,
    "delivery_efficiency_score": delivery_efficiency_score,
    # Target (1)
    "late_delivery": late_delivery
}

df = pl.DataFrame(data_dict)
df.write_csv(OUTPUT_FILE)

print(f"Dataset generated successfully: '{OUTPUT_FILE}' ({df.shape[0]} rows, {df.shape[1]} columns)")
print(f"Late delivery target ratio: {df['late_delivery'].mean():.2%}")