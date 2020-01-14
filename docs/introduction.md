# Estimating Area of Deforestation and Degradation using AREA2 and CODED
![](images/area2.png)

By Eric Bullock (bullocke@bu.edu)

---


This tutorial will demonstrate how the Accuracy and Area Estimation Toolbox (AREA2) can be used to estimate map <a class="ui-tooltip" title="A measure of “correctness”; in the context of this document, accuracy expresses the degree to which the map agrees with reality; an estimate of accuracy is the degree to which the map agrees with a sample of reference observations."><span style="cursor: help;"><i>accuracy</i></span></a> and disturbance area using a dataset created with the Continuous Degradation Detection ([CODED](https://coded.readthedocs.io/en/latest/))<sup>1</sup> methodology on the Google Earth Engine ([GEE](https://earthengine.google.com/))<sup>2</sup>. The equations used in this exercise are presented in Cochran 1977<sup>3</sup> and described in a remote sensing context in Olofsson 2014<sup>4</sup>. 

I refer users to the [AREA2 documentation](https://area2.readthedocs.io/en/latest/overview.html) for background information on the rational of using <a class="ui-tooltip" title="In a sampling framework, an inference expresses the relationship between the population parameter, 𝜇, and its estimate, 𝜇̂ , in probabilistic terms, typically in the form of either of a confidence interval or a test of hypothesis (Dawid, 1983)."><span style="cursor: help;"><i>statistical inference</i></span></a> for assessing map accuracy and area estimation. Definitions for select statistical terminology is provided for words in <i>italics</i> and a more detailed description of terms can be found in the AREA2 documentation. 

### Disclaimers:

1. The data presented is here for the purpose of demonstration. Therefore, it should not be considered reflective of the actual land change dynamics of the region. 
2. All calculations should be validated using the original calculations, and it is the user's responsibility to ensure the calculations produced using AREA2 are correct.  

## Objectives and Tutorial Parts:

1. [Part 1: CODED](coded.md) Create a spatial dataset of deforestation, degradation, forest, and non-forest in the Brazilian Amazon using CODED. 
2. [Part 2: Sample Design and Interpretation](sample.md) Create a sample under stratified random sampling and assign reference labels using the AREA2 toolbox.
3. [Part 3: Estimation of Activity Data and Accuracy](estimation.md) Use AREA2 to estimate areas of activity data and map accuracies.
