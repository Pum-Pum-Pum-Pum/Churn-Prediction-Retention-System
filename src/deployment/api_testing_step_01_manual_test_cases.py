
import json

print('=== API TESTING STEP 1: MANUAL TEST CASES ===')

print('Base URL: http://127.0.0.1:8000')
print('Swagger UI: http://127.0.0.1:8000/docs')

valid_payload = {
    "tenure_months": 12,
    "monthly_charges": 85.5,
    "total_charges_clean": 1026.0,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "Yes",
    "device_protection": "No",
    "tech_support": "No",
    "payment_method": "Electronic check",
    "paperless_billing": 1,
    "partner": 0,
    "dependents": 0,
    "senior_citizen": 0
}

missing_field_payload = {
    "tenure_months": 12,
    "monthly_charges": 85.5,
    "contract": "Month-to-month"
}

wrong_type_payload = {
    "tenure_months": "twelve",
    "monthly_charges": "eighty five",
    "total_charges_clean": 1026.0,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "Yes",
    "device_protection": "No",
    "tech_support": "No",
    "payment_method": "Electronic check",
    "paperless_billing": 1,
    "partner": 0,
    "dependents": 0,
    "senior_citizen": 0
}

extra_field_payload = {
    **valid_payload,
    "unexpected_field": "should_fail_or_be_rejected"
}

print('=== TEST CASE 1: HEALTH CHECK ===')
print('Method: GET')
print('URL: /health')
print('Expected: 200 OK with {"status": "ok"}')

print('=== TEST CASE 2: VALID PREDICTION REQUEST ===')
print('Method: POST')
print('URL: /predict')
print('Payload:')
print(json.dumps(valid_payload, indent=2))
print('Expected: 200 OK with model_version, threshold_used, churn_probability, predicted_label')

print('=== TEST CASE 3: MISSING REQUIRED FIELDS ===')
print('Method: POST')
print('URL: /predict')
print('Payload:')
print(json.dumps(missing_field_payload, indent=2))
print('Expected: validation error / 422 response')

print('=== TEST CASE 4: WRONG DATA TYPES ===')
print('Method: POST')
print('URL: /predict')
print('Payload:')
print(json.dumps(wrong_type_payload, indent=2))
print('Expected: validation error / 422 response')

print('=== TEST CASE 5: EXTRA FIELD ===')
print('Method: POST')
print('URL: /predict')
print('Payload:')
print(json.dumps(extra_field_payload, indent=2))
print('Expected: depends on validation config; ideally reject unexpected fields for stricter API behavior')

print('=== WHAT TO CHECK MANUALLY ===')
