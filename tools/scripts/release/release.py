import subprocess
import sys

import semver
import yaml
from git import Repo

target_version = sys.argv[1]

print(f"Release version: {target_version}")

repo = Repo.init(".")

tags = sorted(repo.tags, key=lambda t: t.commit.committed_datetime)
latest_tag = tags[-1]
latest_tag = latest_tag.name.strip("v")

print(latest_tag)

tv = semver.VersionInfo.parse(target_version)
cv = semver.VersionInfo.parse(latest_tag)

if tv == cv.bump_patch():
    print("Patch version")
elif tv == cv.bump_minor():
    print("Minor version")
elif tv == cv.bump_major():
    print("Major version")
else:
    print("Cannot release - invalid version")
    sys.exit(1)

if repo.active_branch.name != "main":
    print("Not on main branch")
    sys.exit(1)

print("Ensuring origins are up to date")
o = repo.remotes.origin
o.pull()

with open("api/R/cellxgene.census/DESCRIPTION") as f:
    desc = f.read()
    r_desc = yaml.safe_load(desc)
    r_version = r_desc["Version"]
    if r_version != target_version:
        print(f"R version does not match target version: {r_version}")
        sys.exit(1)

# Tag creation

tag = f"v{target_version}"
print("Creating tag: ", tag)
new_tag = repo.create_tag(tag, message=f"Release {target_version}")

print("Pushing tag to origin")
repo.remotes.origin.push(new_tag.name)
print("Tag pushed to origin")

print("Triggering build")
subprocess.run(["gh", "workflow", "run", "py-build.yml", "--ref", tag])

print("Go to https://github.com/chanzuckerberg/cellxgene-census/releases/new to create a release.")

# subprocess.run(["python", "-m", "venv", f"venv-release-{target_version}"])
# subprocess.run([f"venv-release-{target_version}/bin/pip", "install", "cellxgene_census"])
# proc = subprocess.run([f"venv-release-{target_version}/bin/python", "-c", '"import cellxgene_census; print(cellxgene_census.__version__)"'], check=True, capture_output=True)
# print(proc, proc.stdout)
