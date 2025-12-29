include {
    grid_search;
    import_features;
    split_dataset;
} from '../modules/grid_search.nf'
include {
    scatterplot;
    roc_auc_curve;
} from '../modules/visualization.nf'


workflow grid_search_workflow {
    take:
    dataset
    script_grid_search_classification
    script_grid_search_regression
    script_scatterplot
    script_roc_auc_curve
    main:
    algorithms = ["ridge", "lasso", "linear", "mlp"]
    if (params.task == "classification"){
        grid_search(dataset, script_grid_search_classification, algorithms)
        //roc_auc_curve(grid_search.out.test_predictions, script_roc_auc_curve)
    }
    else {
        grid_search(dataset, script_grid_search_regression, algorithms)
        //scatterplot(grid_search.out.test_predictions, script_scatterplot)
    }
    emit:
    cv_results = grid_search.out.cv_results
    test_metrics = grid_search.out.test_metrics
    
}