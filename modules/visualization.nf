process boxplot {
    publishDir "${params.outdir}/plots/", mode: "copy"
    input:
    path(test_metrics_files)
    path(script_boxplot)
    output:
    path("boxplot.png")
    script:
    """
    Rscript ${script_boxplot} ./
    """
    stub:
    """
    touch boxplot.png
    """
}

process scatterplot {
    publishDir "${params.outdir}/plots/", mode: 'copy'
    input:
    path(test_predictions)
    path(script)
    output:
    path("${test_predictions}.scatterplot.png"), emit: scatterplot
    script:
    """
    Rscript ${script} $test_predictions
    cp scatterplot.png ${test_predictions}.scatterplot.png
    """
    stub:
    """
    touch ${test_predictions}.scatterplot.png
    """
}

process roc_auc_curve {
    publishDir "${params.outdir}/plots/", mode: 'copy'
    input:
    path(test_predictions)
    path(script_roc_auc_curve)
    output:
    path("${test_predictions}.roc_auc_curve.png"), emit: roc_auc_curve
    script:
    """
    Rscript ${script_roc_auc_curve} $test_predictions
    cp roc_auc_curve.png ${test_predictions}.roc_auc_curve.png
    """
    stub:
    """
    touch ${test_predictions}.roc_auc_curve.png
    """
}