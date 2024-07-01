# CZ CELLxGENE Discover Census storage & release policy

:exclamation: **This document is for internal use only. Its contents may change without notice.**

**Last edited**: April, 2023.

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "NOT RECOMMENDED" "MAY", and "OPTIONAL" in this document are to be interpreted as described in [BCP 14](https://tools.ietf.org/html/bcp14), [RFC2119](https://www.rfc-editor.org/rfc/rfc2119.txt), and [RFC8174](https://www.rfc-editor.org/rfc/rfc8174.txt) when, and only when, they appear in all capitals, as shown here.

## Definitions

* **Census build**: a SOMA collection with the Census data [as specified in the Census schema](https://github.com/chanzuckerberg/cell-census/blob/main/docs/cell_census_schema.md#data-encoding-and-organization).
* **Census source H5AD files**: the set of H5AD files used to create a Census build.
* **Census release**: a Census build that is publicly hosted online.
* **Census release `tag`**:  a label for a Census release, it MUST be a string of printable ASCII characters.

## Census data storage policy

The following S3 bucket MUST be used as the root to store Census data:

`s3://cellxgene-data-public/`

Census data MUST be deposited under a folder named `cell-census` of the root S3 bucket:

 `./cell-census`

All data related to a Census **release** MUST be deposited in a folder named with the **tag** of the release:

 `./cell-census/[tag]/`

The Census **release** MUST be deposited in a folder named `soma`:

`./cell-census/[tag]/soma/`

All Census **source h5ads** used to create a specific Census **release** MUST be copied into a folder named `h5ads`:
`./cell-census/[tag]/h5ads/`

## Census release information `json`

The publication date along with the full URI paths for the `soma` folder and the `h5ads` folder for all Census releases MUST be recorded in a `json` file with the following naming convention and structure, which will be used as a machine- and human-readable directory of available Census builds:

`./cell-census/releases.json`

* This file MUST be in `json` formats where the parent keys are release identifiers (alias or name).
* The alias `latest` MUST be present and MUST point to the **Weekly Census release**.
* For a long-term supported ("LTS") release:
  * A `flags` key MUST be present with a dictionary-typed value, which MUST contain the key `lts` with the value `true`.
  * An alias MUST exist that points to the release, formatted as `V<#>`, where `<#>` is integer  to label long-term supported Census releases, e.g. `V1`.
* If a long-term-supported release is no longer supported:
  * The `flags` dictionary value MUST contain the key `retracted` with the value `true` and,
  * A `retraction` key MUST be present with a dictionary-typed value, which MUST contain the key `date` with a string value formatted as `<YYYY>-<MM>-<DD>` where `<YYYY>` is the year, `<MM>` is the month, and `<DD>` is the day of the retraction.
  * The `retraction` dictionary value MAY contain:
    * A `reason` key with a string value describing the reason for the retraction.
    * A `info_permalink` key with a string value describing the reason for the retraction.
    * A `replaced_by` key with a string value identifying the alias of a later LTS.

```json
{
   [release_alias]: [release_name|release_alias],
   [release_name]: { #defines a given release
      “release_date”: [yyyy-mm-dd]  #optional, ISO 8601 date, may be null
      “release_build”: [yyyy-mm-dd] #required, ISO 8601 date, date of Census build
      “soma”: {
         “uri”: [uri] #URI of top-level SOMA collection
         “s3_region”: [s3_region] #optional, S3 region if uri is s3://…
      },
      “h5ads”: {
         “uri”: [uri] #base URI of H5AD location
         “s3_region”: [region] #optional, S3 region if uri is s3://…
      },
      “flags”: {
         “lts”: {true|false} # optional, if false
         “retracted”: {true|false} # optional, false
      }
      “retraction”: { # required if "flags" contains "retracted": true, otherwise must not be present
         “date”: [yyyy-mm-dd] # required, ISO 8601 date, date of retraction
         “reason”: [text] # optional, text describing reason for retraction
         “info_permalink”: [url] # optional, URL to more information about retraction
         “replaced_by”: [release_alias] # optional, alias of release that replaces this one
      }
      
   },
   ...
}
```

An example of this file is shown below:

```json
{
   "latest": "release-1"
   "release-1": {
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
   "pre-release-weekly-build-2023-07-15": {
      ...
   },
   ...
}
```
