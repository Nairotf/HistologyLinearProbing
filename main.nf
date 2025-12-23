include {
    import_features;
} from './modules/grid_search.nf'
include {
    grid_search_workflow
} from './workflows/grid_search.nf'
include {
    summary_plot
} from './workflows/visualization.nf'

workflow {
    dataset = Channel.value(file(params.dataset))
    feature_extractors = Channel.fromPath("${projectDir}/params/feature_extractors.csv")
        .splitCsv(header: true, sep: ',')
        .map { row ->
            tuple(
                row.patch_encoder,
                row.slide_encoder,
                row.patch_size,
                row.mag,
                row.batch_size,
                row.overlap
            )
        }
    feature_paths = feature_extractors.map { row ->
        tuple( row[1], file("${params.features_dir}/${row[3]}x_${row[2]}px_${row[5]}px_overlap/slide_features_${row[1]}/"))
    }
    script_import_features = Channel.value(file("${projectDir}/bin/import_features.py"))
    script_grid_search_classification = Channel.value(file("${projectDir}/bin/grid_search_classification.py"))
    script_grid_search_regression = Channel.value(file("${projectDir}/bin/grid_search_regression.py"))
    script_scatterplot = Channel.value(file("${projectDir}/bin/scatter.R"))
    script_roc_auc_curve = Channel.value(file("${projectDir}/bin/roc_curve.R"))
    script_boxplot_r2 = Channel.value(file("${projectDir}/bin/boxplot_r2.R"))
    script_boxplot_auc = Channel.value(file("${projectDir}/bin/boxplot_auc.R"))

    import_features(dataset, params.target, feature_paths, script_import_features)
    grid_search_workflow(import_features.out.dataset,
        script_grid_search_classification, script_grid_search_regression,
        script_scatterplot, script_roc_auc_curve)
    summary_plot(grid_search_workflow.out.cv_results.collect(),
        script_boxplot_r2, script_boxplot_auc)
}