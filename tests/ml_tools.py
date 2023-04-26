import awkward as ak
import dask_awkward as dak
import numpy as np

import coffea.ml_tools


def prepare_jets_array():
    # Creating jagged Jet-with-constituent array
    NJETS = 2000
    NFEAT = 100
    jets = ak.zip(
        {
            "pt": ak.from_numpy(np.random.random(size=NJETS)),
            "eta": ak.from_numpy(np.random.random(size=NJETS)),
            "phi": ak.from_numpy(np.random.random(size=NJETS)),
            "ncands": ak.from_numpy(np.random.randint(1, 50, size=NJETS)),
        },
        with_name="LorentzVector",
    )
    pfcands = ak.zip(
        {
            "pt": ak.from_regular(np.random.random(size=(NJETS, NFEAT))),
            "eta": ak.from_regular(np.random.random(size=(NJETS, NFEAT))),
            "phi": ak.from_regular(np.random.random(size=(NJETS, NFEAT))),
            "feat1": ak.from_regular(np.random.random(size=(NJETS, NFEAT))),
            "feat2": ak.from_regular(np.random.random(size=(NJETS, NFEAT))),
        },
        with_name="LorentzVector",
    )

    idx = ak.local_index(pfcands.pt, axis=-1)
    pfcands = pfcands[idx < jets.ncands]
    jets["pfcands"] = pfcands[:]

    ak_jets = jets[:]
    ak.to_parquet(jets, "ml_tools.parquet")
    dak_jets = dak.from_parquet("ml_tools.parquet")
    return ak_jets, dak_jets


def common_awkward_to_numpy(jets):
    def my_pad(arr):
        return ak.fill_none(ak.pad_none(arr, 100, axis=1, clip=True), 0.0)

    fmap = {
        "points__0": {
            "deta": my_pad(jets.eta - jets.pfcands.eta),
            "dphi": my_pad(jets.phi - jets.pfcands.phi),
        },
        "features__1": {
            "dr": my_pad(
                np.sqrt(
                    (jets.eta - jets.pfcands.eta) ** 2
                    + (jets.phi - jets.pfcands.phi) ** 2
                )
            ),
            "lpt": my_pad(np.log(jets.pfcands.pt)),
            "lptf": my_pad(np.log(jets.pfcands.pt / ak.sum(jets.pfcands.pt, axis=-1))),
            "f1": my_pad(np.log(jets.pfcands.feat1 + 1)),
            "f2": my_pad(np.log(jets.pfcands.feat2 + 1)),
        },
        "mask__2": {
            "mask": my_pad(ak.ones_like(jets.pfcands.pt)),
        },
    }

    return {
        k: ak.concatenate(
            [x[:, np.newaxis, :] for x in fmap[k].values()], axis=1
        ).to_numpy()
        for k in fmap.keys()
    }


def triton_testing():
    # Defining custom wrapper function with awkward padding requirements.
    class triton_wrapper_test(coffea.ml_tools.triton_wrapper):
        def awkward_to_numpy(self, output_list, jets):
            return [], {
                "output_list": output_list,
                "input_dict": common_awkward_to_numpy(jets),
            }

        def dask_touch(self, output_list, jets):
            jets.eta.layout._touch_data(recursive=True)
            jets.phi.layout._touch_data(recursive=True)
            jets.pfcands.pt.layout._touch_data(recursive=True)
            jets.pfcands.phi.layout._touch_data(recursive=True)
            jets.pfcands.eta.layout._touch_data(recursive=True)
            jets.pfcands.feat1.layout._touch_data(recursive=True)
            jets.pfcands.feat2.layout._touch_data(recursive=True)
            pass

    # Running the evaluation in lazy and non-lazy forms
    tw = triton_wrapper_test(
        model_url="triton+grpc://triton.apps.okddev.fnal.gov:443/emj_gnn_aligned/1"
    )

    ak_jets, dak_jets = prepare_jets_array()

    # Numpy arrays testing
    np_res = tw(["softmax__0"], common_awkward_to_numpy(ak_jets))
    print({k: v.shape for k, v in np_res.items()})

    # Vanilla awkward arrays
    ak_res = tw(["softmax__0"], ak_jets)
    print({k: v.to_numpy().shape for k, v in ak_res.items()})

    for k in np_res.keys():
        assert np.all(ak.to_numpy(ak_res[k]) == np_res[k])

    # dask awkward arrays (with lazy_evaluations)
    dak_res = tw(["softmax__0"], dak_jets)
    print({k: v.compute().to_numpy().shape for k, v in dak_res.items()})

    for k in ak_res.keys():
        assert ak.all(ak_res[k] == dak_res[k].compute())
    print(dak.necessary_columns(dak_res))


if __name__ == "__main__":
    triton_testing()