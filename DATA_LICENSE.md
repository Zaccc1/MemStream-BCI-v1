# Data attribution and permitted use

The real Neuropixels example is from the International Brain Laboratory (IBL)
Brain-Wide Map dataset.

- Session EID: `9b5a1754-ac99-4d53-97d3-35c2f6638507`.
- Probe insertion PID: `c2ebf259-9777-4a8f-b1de-5302776961d2`, probe00.
- Session: mainenlab / ZFM-01936 / 2021-01-22 / 002.
- AP dataset UUID: `b94762d8-3634-4a61-8ca8-40eac49aad50`.
- Original dataset MD5: `855488fe222b3581161306ef7c51ac5f`.
- Data provider: International Brain Laboratory, https://www.internationalbrainlab.com/data
- Dataset information: https://figshare.com/articles/preprint/Data_release_-_Brainwide_map_-_Q4_2022/21400815
- License: Creative Commons Attribution 4.0 International,
  https://creativecommons.org/licenses/by/4.0/
- Public data listing: https://registry.opendata.aws/ibl-brain-wide-map/

The packaged AP example is a crop of 240-242 s with the sync channel removed;
the 384 AP channels follow the original pipeline's SpikeGLX reader order.
ADC values and voltage conversion factors are preserved. The downloadable
prefix is the original compressed data restricted to the first ten minutes
(rounded up to its compression boundary); its metadata is adjusted to that
prefix. Neither operation denoises, synthesizes, or changes the neural signal.

Fitted calibration bundles and the KS4 teacher are derived analytical outputs
from 0-240 s of the same recording. Their provenance is provided separately.
Please retain IBL attribution and indicate these transformations when reusing
the example. Do not imply endorsement by IBL. Code has a separate MIT license;
that software license does not replace the example data's CC BY license.

Third-party packages are installed separately and retain their own licenses.
They are not relicensed by this ZIP. The optional GUI, KS4, PyTorch and ONNX
components are subject to their respective distribution terms.
