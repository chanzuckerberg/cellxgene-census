# Cell Census data storage & release policy

**Last edited**: March, 2023.

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "NOT RECOMMENDED" "MAY", and "OPTIONAL" in this document are to be interpreted as described in [BCP 14](https://tools.ietf.org/html/bcp14), [RFC2119](https://www.rfc-editor.org/rfc/rfc2119.txt), and [RFC8174](https://www.rfc-editor.org/rfc/rfc8174.txt) when, and only when, they appear in all capitals, as shown here.

## Definitions

* **Cell Census build**: a SOMA collection with the Cell Census data [as specified in the Cell Census schema](https://github.com/chanzuckerberg/cell-census/blob/main/docs/cell_census_schema.md#data-encoding-and-organization). 
* **Cell Census source h5ads**: the set of h5ad files used for create a Cell Census build.
* **Cell Census release**: a Cell Census build that is officially and publicly hosted online.
* **Cell Census release `tag`**:  a label for a Cell Census release, it MUST be a string of printable ASCII characters.

## Cell Census release policy

### Long-term supported Cell Census release

* It MUST be created and made available for public access every six months. 
* Its date of publication MUST be used as the `release_date` (see [below](#Cell-Census-release-information)). 
* All data from CELLxGENE Discover hosted at the `release_date` that meets the requirements [according to the schema](https://github.com/chanzuckerberg/cell-census/blob/main/docs/cell_census_schema.md#data-included) MUST be included in it.
* It MUST be available for public access for at least 5 years upon `release_date`.

### Weekly Cell Census release (latest)

* It MUST be created and made available for public access every week on Mondays except for CZI holidays.
* Its date of publication MUST be used as the `release_date` (see [below](#Cell-Census-release-information)). 
* All data from CELLxGENE Discover hosted at the `release_date` that meets the requirements [according to the schema](https://github.com/chanzuckerberg/cell-census/blob/main/docs/cell_census_schema.md#data-included) MUST be included in it.
* It MUST be deleted the following Monday and after the next weekly Cell Census release is published.


## Cell Census data storage policy

The following S3 bucket MUST be used as the root to store Cell Census data:

`s3://cellxgene-data-public/`

Cell Census data MUST deposited under a folder named `cell-census` of the root S3 bucket:
 
 `./cell-census`
 
All data related to a Cell Census **release** MUST be deposited in a folder named with the **tag** of the release:

 `./cell-census/[tag]/`


The Cell Census **release** MUST be deposited in a folder named `soma`:

`./cell-census/[tag]/soma/`

All Cell Census **source h5ads** used to create a specific Cell Census **release** MUST be copied into a folder named `h5ads`:	
`./cell-census/[tag]/h5ads/`


## Cell Census release information `json`

The publication date along with the full URI paths for the `soma` folder and the `h5ads` folder  for all Cell Census releases  MUST be recorded in a `json` file with the following naming convention and structure, which will be used as a machine- and human-readable directory of available census builds:

`./cell-census/releases.json`

* This file MUST be in `json` formats where the parent keys are release identifiers (alias or name). 
* The alias `"latest"` MUST be present and MUST point to the **Weekly Cell Census release**. 
* The prefix `"V"` MUST be used followed by an integer counter to label long-term supported Cell Census releases, e.g. `V1`.
* This `json` file MUST follow this schema:


```
{
   [release_alias]: [release_name|release_alias],
   [release_name]: {	#defines a given release
      “release_date”: [yyyy-mm-dd]  #required, ISO 8601 date, may be null
      “release_build”: [yyyy-mm-dd] #required, ISO 8601 date, date of census build
      “soma”: {
         “uri”: [uri] #URI of top-level SOMA collection
         “s3_region”: [s3_region] #optional, S3 region if uri is s3://…
      },
      “h5ads”: {
         “uri”: [uri] #base URI of H5AD location
         “s3_region”: [region] #optional, S3 region if uri is s3://…
      }
   },
   ...
}
```

An example of this file is shown below:

```
{
   "latest": "release-1"
   "V1": 
   "2023-07-15": {
      "release_date": "2023-06-12”,
      "release_build": "2023-05-15”,
      "soma": {
         "uri": "s3://cellxgene-data-public/cell-census/release-1/soma/",
         "s3_region": "us-west-2"
      },
      "h5ads": {
         "uri": "s3://cellxgene-data-public/cell-census/release-1/h5ads/",
         "s3_region": "us-west-2"
      }
   },
   "2023-08-15": {
      ...
   },
   ...
}
```

