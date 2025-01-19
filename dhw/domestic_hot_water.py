def dhw_profile(
    occupancy_distribution,
    daily_amount,
    random_factor,
    active_hours,
    min_large,
    max_large,
    min_draws,
    min_lt,
    max_lt,
    **kwargs
):
    import numpy as np
    from utils import misc

    if kwargs:
        occupancy_distribution = kwargs.get(
            "occupancy_distribution", occupancy_distribution
        )
        daily_amount = kwargs.get("daily_amount", daily_amount)
        random_factor = kwargs.get("random_factor", random_factor)
        active_hours = kwargs.get("active_hours", active_hours)
        min_large = kwargs.get("min_large", min_large)
        max_large = kwargs.get("max_large", max_large)
        min_draws = kwargs.get("min_draws", min_draws)
        min_lt = kwargs.get("min_lt", min_lt)
        max_lt = kwargs.get("max_lt", max_lt)

    regular_draw_amount = np.random.uniform(min_lt, max_lt, size=active_hours)

    # Generate total daily water draw. It is randomised based on the randomisation factor selected above.
    daily_amount += np.random.uniform(-random_factor, random_factor) * daily_amount

    # Generate time at which draw happens. This line insures that the draw timing only
    # occurs when someone is at home
    draw_times = (
        np.random.randint(2, size=occupancy_distribution.shape) * occupancy_distribution
    )
    # print(np.count_nonzero(draw_times))
    if np.count_nonzero(draw_times) < min_draws:
        draw_times = misc.safe_min_ones(occupancy_distribution, draw_times, min_draws)

    # this randomises when the large amount of water (shower or bath) occurs during
    # the day. It can only happen when the person is at home or awake.
    time_large = np.random.choice(np.nonzero(draw_times)[0])
    amount_large = np.random.randint(min_large, max_large)

    # the first line puts the amount of dhw at the correct time defined by "draw_times".
    # The second line will take the randomly chosen hour to have a shower and put that amount of water there
    # because of how it is made it might "override" a regular draw amount
    draw_amounts = draw_times * regular_draw_amount
    draw_amounts[time_large] = amount_large

    # normalize the draw amounts to match the total_daily_water_draw
    draw_amounts = draw_amounts / np.sum(draw_amounts) * daily_amount
    return draw_amounts, draw_times


def dhw_year_day(occupancy_profile_day):
    import numpy as np
    from utils import misc

    dhw_year_daily = []
    dhw_year_daily_times = []
    for element in occupancy_profile_day:
        parameters = misc.dhw_input_generator(element)
        daily_dhw, daily_times = dhw_profile(**parameters)
        dhw_year_daily.append(daily_dhw)
        dhw_year_daily_times.append(daily_times)
    return dhw_year_daily, dhw_year_daily_times
