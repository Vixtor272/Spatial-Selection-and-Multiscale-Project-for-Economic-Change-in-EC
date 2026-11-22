"""
ecuador_price
=============
Code behind the thesis "Selection and Transmission in Space: A Multiscale
Price-Equation Decomposition of Ecuador's Formal Economy".

    paths            every folder and input file, named once
    style            every colour, the map scale and the fonts, named once
    build_db/        raw SRI files -> the analysis table (parish x CIIU section x month)
    decomposition    the driver: prepares the panel and calls the engine of
                     Kemp et al. (UrbanTraitSelection/calculation_scripts/runPrice.py)
    analysis/        small tables behind the text (trait split, robustness, outliers)
    plots/           figure suite, ported from UrbanTraitSelection/analysis_scripts
    videos/          the Loja zoom animations
"""
