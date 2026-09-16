"""Shared robot-control RealSense filters. Import the SDK only at capture time."""
SETTINGS = {
    'order': ['hole_filling', 'spatial', 'temporal'],
    'hole_filling': 'SDK default',
    'spatial': {'filter_magnitude': 5, 'filter_smooth_alpha': .75,
                'filter_smooth_delta': 1, 'holes_fill': 4},
    'temporal': {'filter_smooth_alpha': .75, 'filter_smooth_delta': 1},
}


def make_filters(rs):
    hole = rs.hole_filling_filter()
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    for name, obj in [('spatial', spatial), ('temporal', temporal)]:
        for option, value in SETTINGS[name].items():
            obj.set_option(getattr(rs.option, option), value)
    return [hole, spatial, temporal]
