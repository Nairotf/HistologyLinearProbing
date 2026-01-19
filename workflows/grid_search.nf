include {
    grid_search;
    import_features;
    split_dataset;
} from '../modules/grid_search.nf'
include {
    roc_auc_curve;
} from '../modules/visualization.nf'


workflow grid_search_workflow {
    take:
    dataset
    script_grid_search_classification
    script_roc_auc_curve
    main:
    algorithms = ["elasticnet"]
    grid_search(dataset, script_grid_search_classification, algorithms)
    roc_auc_curve(grid_search.out.test_predictions, script_roc_auc_curve)
    emit:
    cv_results = grid_search.out.cv_results
    test_metrics = grid_search.out.test_metrics
    
}