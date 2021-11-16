import os
import sys
import os.path as osp

ROOT_DIR = osp.join(osp.dirname(osp.dirname(__file__)))
PRIMITIVE_DEMONSTRATIONS_PATH = osp.join(ROOT_DIR, "collect_primitives/")
POLICIES_PATH = osp.join(ROOT_DIR, "policies/")