def build_vle_features(student_vle, vle):
    merged = student_vle.merge(vle, on="id_site", how="left")

    print("Merged columns:", merged.columns.tolist())

    vle_features = (
        merged
        .groupby(["id_student", "code_module"], as_index=False)
        .agg(
            total_clicks=("sum_click", "sum")
        )
    )

    return vle_features
