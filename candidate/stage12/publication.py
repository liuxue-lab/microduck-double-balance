"""Accept only the frozen Stage11 tag or the reviewed documentation followup."""
from .contracts import BASE_COMMIT, TAG_OBJECT, require

REVIEWED_HEAD = "73ee01930a031d76a9d57419afdfd4464a547ab3"
ORIGIN = "git@github.com:liuxue-lab/microduck-double-balance.git"


def validate_publication(observed, review):
    require(review["base"] == review["parent"] == review["tag_target"] == BASE_COMMIT and
            review["reviewed_head"] == REVIEWED_HEAD and review["tag_object"] == TAG_OBJECT,
            "Publication review identity differs")
    require(observed["head"] in (BASE_COMMIT, REVIEWED_HEAD) and
            observed["tag_commit"] == BASE_COMMIT and observed["tag_object"] == TAG_OBJECT and
            observed["branch"] == "double-balance" and observed["origin"] == ORIGIN,
            "Local code/tag identity differs from the two reviewed commits; inspect identity-observed JSON")
    if observed["head"] == REVIEWED_HEAD:
        expected = {r["path"]: {"status": {"added": "A", "modified": "M"}[r["status"]],
                                "git_blob_sha1": r["git_blob_sha1"]} for r in review["files"]}
        require(observed["parent"] == BASE_COMMIT and observed["documentation_changes"] == expected,
                "Reviewed documentation followup parent/file/blob identity differs")
