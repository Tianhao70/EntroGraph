from __future__ import annotations


class HallusionDataset:
    """P2 skeleton for HallusionBench.

    HallusionBench is primarily yes/no visual illusion and language
    hallucination evaluation. The full loader is intentionally left as a
    follow-up so the CHAIR/MMHal open-ended path stays focused.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("HallusionBench loader is a P2 skeleton for now.")

