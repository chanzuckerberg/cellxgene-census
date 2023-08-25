## New Features in the CZ CELLxGENE Census API

**Published September 2023**

To our community of computational biologists, software engineers, and machine learning research scientists, we are pleased to announce the upcoming improvements to the CZ CELLxGENE Census API. The following details highlight the enhancements:

### 1. **Numerical Stability in the X[normalized] Layer**

The X[normalized] layer has undergone precision refinement. Previously, minute fractional values in this layer were written as explicit zeros. In the updated version, a small sigma value is consistently added to each position in the layer. This assures that all explicitly documented coordinates possess values greater than zero, adhering to the principle that a normalized layer should never record a zero for a corresponding value of 1 in the raw layer. For further technical details, refer to issue [#706](https://github.com/chanzuckerberg/cellxgene-census/issues/706).

### 2. **Data Type Upgradation for Enhanced Accuracy**

To ensure optimal accuracy, the `obs.raw_sum`, `obs.raw_mean_nnz`, and `obs.raw_variance_nnz` fields have transitioned from `float32` to `float64` data type. This change primarily addresses the concern of insufficient precision offered by `float32` in recording axis statistics. For a comprehensive understanding, refer to the associated issue [#714](https://github.com/chanzuckerberg/cellxgene-census/issues/714).

### 3. **Introduction of Summary Statistics**

Summary statistics have been incorporated to facilitate swift and precise data interpretation:

- For `census["census_data"][organism].obs`:
  * `raw_sum` - Represents the count sum derived from X["raw"]
  * `nnz` - Enumerates the number of non-zero (nnz) values
  * `raw_mean` - Provides the average count of nnz values
  * `raw_variance` - Indicates the count variance of nnz values
  * `n_measured_vars` - Enumerates the "measured" genes, determined by the sum of the presence matrix

It is noteworthy that the metric detailing the number of mitochondrial genes has been excluded due to inconsistencies in its biological accuracy across diverse datasets.

- For `census["census_data"][organism].ms["RNA"].var`:
  * `nnz` - Enumerates the number of nnz values
  * `n_measured_obs` - Specifies the number of "measured" cells, as per the sum of the presence matrix

For in-depth technical specifications, please review issues [#289](https://github.com/chanzuckerberg/cellxgene-census/issues/289) and [#563](https://github.com/chanzuckerberg/cellxgene-census/issues/563).

### **Significance**:
- **Count Sum Functionality**: The count sum is a fundamental aspect of normalization processes.
- **Mitochondrial Genes Metric**: It serves as a quality control metric to assess the likelihood of a cell undergoing apoptosis.

The incorporation of these advanced features substantially streamlines single cell analysis tasks. These enhancements signify our commitment to delivering optimal analytical tools for the research community. We encourage users to explore and familiarize themselves with these new additions for maximum utility in their scientific pursuits.
