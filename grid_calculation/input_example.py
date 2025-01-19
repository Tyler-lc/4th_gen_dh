from calculate_grid_layout_V2 import run_create_network

input_data = {
    "n_supply_list": [
        {
            "id": "S1",
            "coords": [52.5200, 13.4050],  # latitude, longitude
            "cap": 1000,  # capacity in kW
        },
        {"id": "S2", "coords": [52.5180, 13.4030], "cap": 800},
    ],
    "n_demand_list": [
        {"id": "D1", "coords": [52.5220, 13.4070], "cap": 500},
        {"id": "D2", "coords": [52.5190, 13.4060], "cap": 300},
        {"id": "D3", "coords": [52.5210, 13.4040], "cap": 400},
    ],
    "ex_grid_data": [
        {
            "start": [52.5195, 13.4055],
            "end": [52.5205, 13.4065],
            "diameter": 0.2,  # meters
            "length": 500,  # meters
        }
    ],
    "coords_list": [
        [52.5170, 13.4020],
        [52.5230, 13.4020],
        [52.5230, 13.4080],
        [52.5170, 13.4080],
    ],
    "ex_cap": [
        {"number": "S1", "classification_type": "source", "capacity": 1000},
        {"number": "S2", "classification_type": "source", "capacity": 800},
        {"number": "D1", "classification_type": "sink", "capacity": 500},
        {"number": "D2", "classification_type": "sink", "capacity": 300},
        {"number": "D3", "classification_type": "sink", "capacity": 400},
    ],
}

road_nw, n_demand_dict, n_supply_dict = run_create_network(input_data)
