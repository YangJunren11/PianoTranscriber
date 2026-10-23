# The trained network's licence

`BarCounter.mlpackage` (Core ML) and `barcounter.pt` (the same network's PyTorch weights) are
licensed under the
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International licence](https://creativecommons.org/licenses/by-nc-sa/4.0/)
(CC BY-NC-SA 4.0), not the Apache licence that covers the code.

That is because some of what it learned from is licensed that way: ASAP's scores and the AudioLabs v2
measure annotations (see `../NOTICE`). So it may be used, shared and adapted, but not commercially,
and what is made from it has to be shared on the same terms, crediting:

> Grand Staff Bar Counter for PDFs (YangJunren11), trained on scores from PDMX and ASAP and on the
> AudioLabs v2 measure annotations.

For a network free of these terms, train one on the public-domain PDMX scores and the generated text
pages alone (see `../README.md`, "Training"), leaving out ASAP and AudioLabs.
