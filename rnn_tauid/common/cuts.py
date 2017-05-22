def and_cuts(cuts):
    """Combines a list of cut expressions with the logical and"""
    parens = ["({0})".format(cut) for cut in cuts]
    return " && ".join(parens)


mode1P3PWithTruth = "(TauJets.nTracks == 1 || TauJets.nTracks == 3) " \
                    "&& (TauJets.truthProng == 1 || TauJets.truthProng == 3)"
mode1P3PNoTruth = "TauJets.nTracks == 1 || TauJets.nTracks == 3"
eta25 = "abs(TauJets.eta) < 2.5"
eta25Truth = "abs(TauJets.truthEtaVis) < 2.5"
pt20 = "TauJets.pt > 20000"
pt20Truth = "TauJets.truthPtVis > 20000"
vetoCrack = "abs(TauJets.eta) < 1.37 || abs(TauJets.eta) > 1.52"
vetoCrackTruth = "abs(TauJets.truthEtaVis) < 1.37 " \
                 "|| abs(TauJets.truthEtaVis) > 1.52"
matched = "TauJets.IsTruthMatched"
mode1P = "TauJets.nTracks == 1"
mode3P = "TauJets.nTracks == 3"

matchKin = and_cuts([matched, eta25Truth, vetoCrackTruth, pt20Truth])
baseline = and_cuts([eta25, vetoCrack, pt20, matchKin])
baselineNoTruth = and_cuts([eta25, vetoCrack, pt20])

sel_truth_1p = and_cuts([baseline, mode1P3PWithTruth, mode1P])
sel_1p = and_cuts([baselineNoTruth, mode1P3PNoTruth, mode1P])
sel_truth_3p = and_cuts([baseline, mode1P3PWithTruth, mode3P])
sel_3p = and_cuts([baselineNoTruth, mode1P3PNoTruth, mode3P])

# For Upgrade samples
eta40 = "abs(TauJets.eta) < 4.0"
eta40Truth = "abs(TauJets.truthEtaVis) < 4.0"
matchKinUpgrade = and_cuts([matched, eta40Truth, pt20Truth])
baselineUpgrade = and_cuts([eta40, pt20, matchKinUpgrade])
baselineNoTruthUpgrade = and_cuts([eta40, pt20])

sel_truth_Xp = baselineUpgrade
sel_Xp = baselineNoTruthUpgrade

sel_dict = {"truth1p": sel_truth_1p, "1p": sel_1p,
            "truth3p": sel_truth_3p, "3p": sel_3p,
            "truthXp": sel_truth_Xp, "Xp": sel_Xp}
