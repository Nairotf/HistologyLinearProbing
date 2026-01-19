include {
    boxplot;
} from '../modules/visualization.nf'

workflow summary_plot {
    take:
    test_metrics
    script_boxplot_auc
    main:
    boxplot(test_metrics, script_boxplot_auc)
}