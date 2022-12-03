# Cell Census storage & release policy

**Status**: Draft.

**Version**: 0.0.1.

**Last edited**: Dec, 2022.

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "NOT RECOMMENDED" "MAY", and "OPTIONAL" in this document are to be interpreted as described in [BCP 14](https://tools.ietf.org/html/bcp14), [RFC2119](https://www.rfc-editor.org/rfc/rfc2119.txt), and [RFC8174](https://www.rfc-editor.org/rfc/rfc8174.txt) when, and only when, they appear in all capitals, as shown here.

---

The cell census MUST be stored in the following S3 root bucket:

`s3://cellxgene-data-public/`

Each release MUST be deposited in a folder under the root that follows this naming convention. `counter` MUST be an integer indicating the release number starting on `1`:

`./cell-census/release-[counter]/`

The Cell Census SOMA-related files MUST be deposited within the release folder on a folder named `soma`:

`./cell-census/release-[counter]/soma/`

All h5ads used to create the Cell Census MUST be copied within the release folder into a folder named `h5ads`:	
`./cell-census/release-[counter]/h5ads/`

Any dataset changes, additions, or deletions per release MUST be documented in the following human-readable changelog file name `changelog.txt`:

`./cell-census/changelog.txt`

The publication date along with the full URI paths for the `soma` folder and the `h5ads` folder  for all Cell Census releases  MUST be recorded in a `json` file with the following naming convention and structure, which will be used as a machine- and human-readable directory of available census builds:


`./cell-census/releases.json`

This file MUST have the following `json` schema:

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

