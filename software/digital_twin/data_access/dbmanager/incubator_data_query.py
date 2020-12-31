import logging

import pandas

_l = logging.getLogger("IncubatorDataQuery")


def query(query_api, bucket, start_date_ns, end_date_ns, measurement, field):
    _l.debug("Query: ")
    query_to_be = f"""
        from(bucket: "{bucket}")
          |> range(start: time(v: {start_date_ns}), stop: time(v: {end_date_ns}))
          |> filter(fn: (r) => r["_measurement"] == "{measurement}")
          |> filter(fn: (r) => r["_field"] == "{field}")
        """
    _l.debug(query_to_be)
    result = query_api.query_data_frame(query_to_be)
    assert isinstance(result, pandas.DataFrame), "Result is a dataframe"
    _l.debug(f"New query for {measurement}'s {field}: {len(result)} samples retrieved.")
    return result
