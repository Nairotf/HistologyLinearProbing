process import_features {
    publishDir "${params.outdir}/features/", mode: "copy"
    tag "${feature_extractor}"
    input:
        tuple path(dataset), val(target_column), path(splits)
        tuple val(feature_extractor), path(features_dir)
        path(script)
    output:
        tuple path("${feature_extractor}.h5"), val(feature_extractor), path(splits), emit: dataset
    script:
        """
        python -u ${script} $dataset $target_column $features_dir ${feature_extractor}
        """
    stub:
        """
        touch "${feature_extractor}.h5"
        """
}

process split_dataset {
    publishDir "${params.outdir}/splits/", mode:"copy"
    input:
        path(dataset)
        val(target_column)
        path(script)
    output:
        tuple path(dataset), val(target_column), path(target_column), emit: splits
    script:
        """
        python $script --csv_path $dataset --target $target_column --output_name $target_column
        """
    stub:
        """
        mkdir -p splits/$target_column/
        cp $dataset splits/$target_column/
        touch splits_0_bool.csv
        touch splits_1_bool.csv
        touch splits_2_bool.csv
        touch splits_3_bool.csv
        touch splits_4_bool.csv
        """
}

process grid_search {
    publishDir "${params.outdir}/models/", mode: 'copy', pattern: "*.pipeline.joblib"
    publishDir "${params.outdir}/cv_result/", mode: 'copy', pattern: "*.cv_result.csv"
    publishDir "${params.outdir}/test_metrics/", mode: 'copy', pattern: "*.test_metrics.csv"
    publishDir "${params.outdir}/test_predictions/", mode: 'copy', pattern: "*.test_predictions.csv"
    publishDir "${params.outdir}/best_params/", mode: 'copy', pattern: "*.best_params.json"
    tag "${feature_extractor}"
    input:
        tuple path(dataset), val(feature_extractor), path(splits)
        path(script)
        each model
    output:
        path("${feature_extractor}.${model}.cv_result.csv"), emit: cv_results
        path("${feature_extractor}.${model}.test_metrics.csv"), emit: test_metrics
        path("${feature_extractor}.${model}.*.test_predictions.csv"), emit: test_predictions
        path("${feature_extractor}.${model}.pipeline.joblib"), emit: pipeline
        path("${feature_extractor}.${model}.best_params.json"), emit: best_params
    script:
        """
        python -u ${script} $dataset $model $feature_extractor $splits
        """
    stub:
        """
        touch ${feature_extractor}.${model}.cv_result.csv
        touch ${feature_extractor}.${model}.test_metrics.csv
        touch ${feature_extractor}.${model}.test_predictions.csv
        touch ${feature_extractor}.${model}.pipeline.joblib
        touch ${feature_extractor}.${model}.best_params.json
        """
}

process concat_results {
    publishDir "${params.outdir}/test_metrics/", mode:"copy"
    input:
    path(csv)
    output:
    path("summary.csv"), emit: summary
    script:
    """
    head -n 1 ${csv[0]} > head.txt
    cat ${csv} | grep -v "fold" > body.txt
    cat head.txt body.txt > summary.csv
    """
    stub:
    """
    touch summary.csv
    """
}