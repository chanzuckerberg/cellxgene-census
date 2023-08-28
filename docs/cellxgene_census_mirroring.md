# CELLxGENE Census Mirroring

The Census supports geographical mirroring. A list of mirrors is available in the `mirrors.json` file, which can be accessed at <https://census.cellxgene.cziscience.com/cellxgene-census/v1/mirrors.json>.

The `mirrors.json` file will look like this:

```json
{
    "default": "AWS-S3-us-west-2",
    "AWS-S3-us-west-2": {
        "provider": "S3",
        "base_uri": "s3://cellxgene-data-public/",
        "region": "us-west-2"
    }
}
```

Each mirror is a dictionary entry that contains three fields:

* `provider`: A string that identifies a cloud provider for the hosting service, e.g. `S3` or `GCS`. Currently only `S3` is supported.
* `base_uri`: URI that identifies the resource hosting the census content.
* `region`: Geographical region where the resource is located.

When calling `open_soma`, the Census API will use the mirroring information to locate the SOMA artifacts. Remember that SOMA artifacts are defined in `release.json` and each entry looks like this:

```json
"soma": {
    "uri": "s3://cellxgene-data-public/cell-census/2023-07-25/soma/",
    "relative_uri": "/cell-census/2023-07-25/soma/",
    "s3_region": "us-west-2"
},
```

This is what happens under the hood:

* The content of `mirrors.json` is retrieved from Cloudfront.
* If the user specifies a mirror (via `open_soma(mirror=...)`), we check if the mirror exists in the registry.
* If the user does not specify a mirror, the default mirror is selected.
* The census release directory is retrieved
* For the chosen version, the `relative_uri` field in the locator is resolved against the `base_uri` of the chosen mirror.
* This resolved URI is used to access the SOMA artifacts.

Note that Census mirroring is only supported from version 1.6.0 onwards. Previous versions will only be able to access the data from a single location: the one specified in the `uri` field of the locator (together with `s3_region`). These fields could be deprecated in a future release of the Census.
