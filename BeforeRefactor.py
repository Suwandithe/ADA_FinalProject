import requests
import numpy as np
import time
from functools import lru_cache
import folium
from polyline import decode
import heapq  # Priority queue

# API KEY
API_KEY = 'AIzaSyA24dMb2PZRNP-6bA_5ynMm9qJWzcb9cS8'

# Global variable for the distance matrix used in TSP
DIST_MATRIX = None



def get_route_polyline(origin, destination, mode='driving'):
    url = f"https://maps.googleapis.com/maps/api/directions/json"
    params = {
        'origin': f"{origin[0]},{origin[1]}",
        'destination': f"{destination[0]},{destination[1]}",
        'mode': mode,
        'key': API_KEY
    }
    response = requests.get(url, params=params).json()
    if response['status'] == 'OK':
        # Extract the polyline from the first route
        polyline = response['routes'][0]['overview_polyline']['points']
        return decode(polyline)  # Decode polyline into a list of (lat, lng) tuples
    else:
        raise ValueError(f"Error fetching route: {response['status']}")


# Function to get the coordinates of a place using the Google Maps Geocoding API
def get_coordinates(place):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={place}&key={API_KEY}"
    response = requests.get(url).json()
    print(response)
    if response['status'] == 'OK':
        location = response['results'][0]['geometry']['location']
        return (location['lat'], location['lng'])
    else:
        raise ValueError(f"Error finding location: {place}")


# Function to get distance matrix using the Google Maps API
def get_distance_matrix(locations, mode="driving"):
    origin_str = '|'.join([f"{lat},{lng}" for lat, lng in locations])
    url = f"https://maps.googleapis.com/maps/api/distancematrix/json?origins={origin_str}&destinations={origin_str}&mode={mode}&key={API_KEY}"
    response = requests.get(url).json()
    print(response)
    if response['status'] != 'OK':
        raise ValueError("Error in Distance Matrix API call.")
    
    distance_matrix = []
    duration_matrix = []
    
    for row in response['rows']:
        distance_row = [elem['distance']['value'] for elem in row['elements']]
        duration_row = [elem['duration']['value'] for elem in row['elements']]
        distance_matrix.append(distance_row)
        duration_matrix.append(duration_row)
    
    return np.array(distance_matrix), np.array(duration_matrix)  # Return distances and durations in meters and seconds respectively


# # Dijkstra's Algorithm (Greedy approach for nearest neighbor)
def dijkstra_greedy(dist_matrix):
    n = len(dist_matrix)
    visited = [False] * n
    current_location = 0
    visited[current_location] = True
    total_distance = 0
    route = [current_location]
    
    for _ in range(n - 1):
        nearest_dist = float('inf')
        nearest_location = -1
        for i in range(n):
            if not visited[i] and dist_matrix[current_location][i] < nearest_dist:
                nearest_dist = dist_matrix[current_location][i]
                nearest_location = i
        route.append(nearest_location)
        total_distance += nearest_dist
        visited[nearest_location] = True
        current_location = nearest_location
    
    total_distance += dist_matrix[current_location][0]  # Return to the start
    route.append(0)
    
    return route, total_distance


@lru_cache(None)
def tsp(dp_mask, pos):
    global DIST_MATRIX
    if dp_mask == (1 << len(DIST_MATRIX)) - 1:
        return DIST_MATRIX[pos][0]  # Return to starting point
    
    min_distance = float('inf')
    for city in range(len(DIST_MATRIX)):
        if dp_mask & (1 << city) == 0:  # If city is not visited
            new_mask = dp_mask | (1 << city)
            min_distance = min(min_distance, DIST_MATRIX[pos][city] + tsp(new_mask, city))
    
    return min_distance

# Wrapper function to solve TSP with memoization
def solve_tsp(dist_matrix):
    global DIST_MATRIX
    DIST_MATRIX = tuple(map(tuple, dist_matrix))  # Convert NumPy array to tuple of tuples
    initial_mask = 1  # Start from the first city
    return tsp(initial_mask, 0)

# Function to visualize the route on a map using Folium

def visualize_route(locations, route, mode):
    # Create a folium map centered at the first location
    start_location = locations[0]
    route_map = folium.Map(location=start_location, zoom_start=13)
    
    # Add markers for all locations with delivery order
    for order, idx in enumerate(route):
        lat, lng = locations[idx]
        folium.Marker(
            location=(lat, lng),
            popup=f"Location {idx + 1} (Stop {order + 1})",
            icon=folium.DivIcon(html=f'<div style="font-size: 14px; color: blue;">{order + 1}</div>')
        ).add_to(route_map)
    
    # Add polylines for the route
    for i in range(len(route) - 1):
        origin = locations[route[i]]
        destination = locations[route[i + 1]]
        polyline_coords = get_route_polyline(origin, destination)
        folium.PolyLine(polyline_coords, color="blue", weight=5).add_to(route_map)
    
    # Save the map as an HTML file
    map_file = f"optimized_route_{mode.lower().replace(' ', '_')}.html"
    route_map.save(map_file)
    print(f"\n{mode} route map has been saved as {map_file}. Open it in a browser to view.")
    return map_file


def get_user_location_input(locations, location_index):
    while True:
        location_input = input(f"Enter the name or coordinates (lat,lng) of location {location_index + 1}: ")

        # Try to parse the location as coordinates first
        if ',' in location_input:
            try:
                lat, lng = map(float, location_input.split(','))
                new_location = (lat, lng)
            except ValueError:
                print("Invalid coordinates format. Please try again.")
                continue
        else:
            # Otherwise, treat it as a name and fetch coordinates
            try:
                new_location = get_coordinates(location_input)
            except ValueError as e:
                print(e)
                continue

        # Check if the location already exists in the list
        if new_location in locations:
            print("Error: You have already entered this location. Please input a different location.")
        else:
            locations.append(new_location)
            break


def main():
    # Step 1: Get user input for driver's location
    print("Welcome to Route Optimization Project!")
    driver_location_input = input("Enter your current location (lat,lng) or name of the location: ")
    
    # Try to parse the driver's location as coordinates first
    try:
        driver_lat, driver_lng = map(float, driver_location_input.split(','))
        driver_location = (driver_lat, driver_lng)
    except ValueError:
        # If it fails, treat it as a location name and fetch coordinates
        try:
            driver_location = get_coordinates(driver_location_input)
        except ValueError:
            print("Invalid location input. Please try again.")
            return

    # Step 2: Add the driver's location to the list
    locations = [driver_location]  # Start with driver's location

    # Step 3: Get user input for other locations
    n = int(input("Enter the number of other locations you want to visit: "))
    for i in range(n):
        get_user_location_input(locations, i)

    # Step 3: Get the distance and duration matrices from Google Maps API
    dist_matrix, dur_matrix = get_distance_matrix(locations, mode='driving')

    # Step 4: Run Dijkstra's Algorithm (Greedy)
    print("\nRunning Greedy Algorithm (Dijkstra's Approximation)...")
    start_time = time.time()  # Record start time for Greedy Algorithm
    greedy_route, greedy_distance = dijkstra_greedy(dist_matrix)
    greedy_duration = sum(dur_matrix[greedy_route[i]][greedy_route[i + 1]] for i in range(len(greedy_route) - 1))
    greedy_execution_time = time.time() - start_time  # Calculate execution time for Greedy Algorithm

    print(f"Greedy Algorithm Route (Index-Based): {greedy_route}")
    print(f"Total Distance (Greedy): {greedy_distance / 1000:.2f} km")
    print(f"Total Duration (Greedy): {greedy_duration / 60:.2f} minutes")  # Duration in minutes
    print(f"Execution Time (Greedy Algorithm): {greedy_execution_time:.2f} seconds")

    # Visualize Greedy route
    visualize_route(locations, greedy_route, "Greedy")

    # Step 5: Run Dynamic Programming (TSP Solver)
    print("\nRunning Dynamic Programming (TSP Solver)...")
    start_time = time.time()  # Record start time for TSP Solver
    tsp_distance = solve_tsp(dist_matrix)
    tsp_route = list(range(len(locations)))  # Create a dummy route as TSP does not output the path
    tsp_duration = sum(dur_matrix[greedy_route[i]][greedy_route[i + 1]] for i in range(len(greedy_route) - 1))
    tsp_execution_time = time.time() - start_time  # Calculate execution time for TSP Solver

    print(f"Total Distance (TSP - DP): {tsp_distance / 1000:.2f} km")
    print(f"Total Duration (TSP - DP): {tsp_duration / 60:.2f} minutes")  # Duration in minutes
    print(f"Execution Time (TSP Solver): {tsp_execution_time:.2f} seconds")

    # Visualize TSP route (dummy in this example)
    visualize_route(locations, tsp_route, "TSP-DP")
    print("\nRoute optimization completed!")



if __name__ == "__main__":
    main()
