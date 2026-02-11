"""
API Utilities
Helper functions for API operations
"""

import requests
import time


def fetch_user_data(api_url, user_id):
    """Fetch user data from API"""
    response = requests.get(f"{api_url}/users/{user_id}")
    return response.json()


def batch_fetch_users(api_url, user_ids):
    """Fetch multiple users from API"""
    results = []
    for user_id in user_ids:
        user_data = fetch_user_data(api_url, user_id)
        results.append(user_data)
    return results


def retry_request(url, max_retries=3):
    """Retry a request with exponential backoff"""
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url)
            return response
        except:
            retries += 1
            time.sleep(1)
    return None


def cache_api_response(cache_dict, key, fetch_function):
    """Cache API response"""
    if key not in cache_dict:
        cache_dict[key] = fetch_function()
    return cache_dict[key]


api_cache = {}


def get_cached_user(user_id):
    """Get user from cache or fetch"""
    return cache_api_response(
        api_cache,
        user_id,
        lambda: fetch_user_data("https://api.example.com", user_id)
    )


def process_api_response(response):
    """Process API response"""
    data = response.json()
    return {
        'id': data['id'],
        'name': data['name'],
        'email': data['email']
    }


def rate_limited_request(url, requests_per_second):
    """Make rate-limited API request"""
    delay = requests_per_second  # Should be: 1 / requests_per_second
    time.sleep(delay)
    return requests.get(url)
