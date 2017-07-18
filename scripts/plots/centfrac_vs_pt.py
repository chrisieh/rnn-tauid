from argparse import Namespace
from var_vs_pt import main

if __name__ == "__main__":
    args = Namespace(
        data="sig1P_test.root",
        xvar="TauJets.pt",
        yvar="TauJets.centFrac",
        x_range=(20., 100.),
        y_range=(0., 1.),
        num_x_bins=30,
        num_y_bins=30,
        x_log=False,
        y_log=False,
        z_log=False,
        x_scale=1000.0,
        y_scale=None,
        x_label=r"Reconstructed tau $p_\mathrm{T}$ / GeV",
        y_label=r"Central energy fraction $f_\mathrm{cent}$",
        outfile="cent_frac_vs_pt_sig.pdf"
    )
    args2 = Namespace(
        data="bkg1P_weight_test.root",
        xvar="TauJets.pt",
        yvar="TauJets.centFrac",
        x_range=(20., 100.),
        y_range=(0., 1.),
        num_x_bins=40,
        num_y_bins=40,
        x_log=False,
        y_log=False,
        z_log=False,
        x_scale=1000.0,
        y_scale=None,
        x_label=r"Reconstructed tau $p_\mathrm{T}$ / GeV",
        y_label=r"Central energy fraction $f_\mathrm{cent}$",
        outfile="cent_frac_vs_pt_bkg.pdf"
    )

    main(args)
    main(args2)
