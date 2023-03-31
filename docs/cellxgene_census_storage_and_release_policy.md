# Cellxgene Census storage & release policy

**Last edited**: Dec, 2022.

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "NOT RECOMMENDED" "MAY", and "OPTIONAL" in this document are to be interpreted as described in [BCP 14](https://tools.ietf.org/html/bcp14), [RFC2119](https://www.rfc-editor.org/rfc/rfc2119.txt), and [RFC8174](https://www.rfc-editor.org/rfc/rfc8174.txt) when, and only when, they appear in all capitals, as shown here.

---

The cellxgene census MUST be stored in the following S3 root bucket:

`s3://cellxgene-data-public/`

All contents of a Cellxgene Census build data MUST be deposited in a folder named `cell-census`: 

`./cell-census/`

This folder MUST contain all public builds and releases defined as:

* Census "build" is a SOMA collection / container containing cell data.
* Census "RELEASE" is a "build" which has been officially blessed and released in a formal manner.
* A "build" is named with a `tag`, which is a string of printable ASCII (eg, "foo", "2022-11-01", "1-rc2.9"):
   `./cell-census/[tag]/`
* a RELEASE is named with a tag following the conventional format `release-[counter]`:
   `./cell-census/release-[counter]/`

The Cellxgene Census SOMA-related files MUST be deposited within the release folder on a folder named `soma`:

`./cell-census/release-[counter]/soma/`

All h5ads used to create the Cellxgene Census MUST be copied within the release folder into a folder named `h5ads`:	
`./cell-census/release-[counter]/h5ads/`

Any dataset changes, additions, or deletions per release MUST be documented in the following human-readable changelog file name `changelog.txt`:

`./cell-census/changelog.txt`

The publication date along with the full URI paths for the `soma` folder and the `h5ads` folder  for all Cellxgene Census releases  MUST be recorded in a `json` file with the following naming convention and structure, which will be used as a machine- and human-readable directory of available census builds:


`./cell-census/releases.json`

This file MUST be in `json` formats where the parent keys are release identifiers (alias or name). The alias `"latest"` MUST be present. This `json` file MUST follow this schema:


```
{
   [release_alias]: [release_name|release_alias],
   [release_name]: {	#defines a given release
      “release_date”: [yyyy-mm-dd]  #optional, ISO 8601 date, may be null
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

