import math
import copy
import time
import hashlib
from typing import List, Dict, Tuple, Optional, Set, Any
from flask import Flask, request, jsonify, render_template

# ==============================================================================
# 1. GEOMETRIC MODELS & DATA STRUCTURES
# ==============================================================================

class Rectangle:
    __slots__ = ('x', 'y', 'w', 'h')

    def __init__(self, x: float, y: float, w: float, h: float):
        self.x = round(float(x), 4)
        self.y = round(float(y), 4)
        self.w = round(float(w), 4)
        self.h = round(float(h), 4)

    @property
    def width(self) -> float:
        return self.w

    @property
    def height(self) -> float:
        return self.h

    @property
    def x2(self) -> float:
        return round(self.x + self.w, 4)

    @property
    def y2(self) -> float:
        return round(self.y + self.h, 4)

    @property
    def area(self) -> float:
        return round(self.w * self.h, 4)

    def intersects_interior(self, other: 'Rectangle') -> bool:
        """Checks if two rectangles overlap in their strict interior (area intersection)."""
        return not (
            self.x2 <= other.x or
            other.x2 <= self.x or
            self.y2 <= other.y or
            other.y2 <= self.y
        )

    def contains(self, other: 'Rectangle') -> bool:
        """Checks if this rectangle completely contains another."""
        return (
            self.x <= other.x and
            self.y <= other.y and
            self.x2 >= other.x2 and
            self.y2 >= other.y2
        )

    def __repr__(self):
        return f"Rect(x={self.x}, y={self.y}, w={self.w}, h={self.h})"


class PieceInstance:
    def __init__(
        self,
        instance_id: str,
        name: str,
        original_file_name: Optional[str],
        width: float,
        height: float,
        rotation_allowed: bool = True,
        alphabetical_rank: int = 0
    ):
        self.id = instance_id
        self.name = name
        self.original_file_name = original_file_name
        self.original_w = round(float(width), 4)
        self.original_h = round(float(height), 4)
        self.rotation_allowed = rotation_allowed
        self.alphabetical_rank = alphabetical_rank

        # Placement state
        self.x: float = 0.0
        self.y: float = 0.0
        self.rotated: bool = False
        self.state: str = "OUTSIDE"  # "INSIDE" or "OUTSIDE"

    @property
    def effective_w(self) -> float:
        return self.original_h if self.rotated else self.original_w

    @property
    def effective_h(self) -> float:
        return self.original_w if self.rotated else self.original_h

    @property
    def width(self) -> float:
        return self.original_w

    @property
    def height(self) -> float:
        return self.original_h

    @property
    def area(self) -> float:
        return round(self.original_w * self.original_h, 4)

    @property
    def x2(self) -> float:
        return round(self.x + self.effective_w, 4)

    @property
    def y2(self) -> float:
        return round(self.y + self.effective_h, 4)

    def to_rect(self) -> Rectangle:
        return Rectangle(self.x, self.y, self.effective_w, self.effective_h)

    def clone(self) -> 'PieceInstance':
        p = PieceInstance(
            self.id, self.name, self.original_file_name,
            self.original_w, self.original_h, self.rotation_allowed,
            self.alphabetical_rank
        )
        p.x = self.x
        p.y = self.y
        p.rotated = self.rotated
        p.state = self.state
        return p

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "width": self.original_w,
            "height": self.original_h,
            "effective_w": self.effective_w,
            "effective_h": self.effective_h,
            "x": self.x,
            "y": self.y,
            "rotated": self.rotated,
            "state": self.state,
            "originalFileName": self.original_file_name
        }


class Sheet:
    def __init__(self, width: float, height: float):
        self.width = round(float(width), 4)
        self.height = round(float(height), 4)

    @property
    def area(self) -> float:
        return round(self.width * self.height, 4)

    def to_rect(self) -> Rectangle:
        return Rectangle(0.0, 0.0, self.width, self.height)


# ==============================================================================
# 2. ALPHABETICAL & IMPORT MANAGER (REVERSED SORT)
# ==============================================================================

class AlphabeticalManager:
    @staticmethod
    def expand_and_rank_pieces(pieces_def: List[Dict[str, Any]]) -> List[PieceInstance]:
        """
        Sorts items in REVERSE alphabetical order by file name / piece name
        and assigns integer alphabetical_rank.
        """
        sorted_defs = sorted(
            pieces_def,
            key=lambda p: (
                str(p.get('originalFileName') or '').lower(),
                str(p.get('name') or '').lower()
            ),
            reverse=True  # ترتیب معکوس
        )

        all_instances = []
        rank_counter = 0

        for pdef in sorted_defs:
            name = str(pdef.get('name', 'Part'))
            w = float(pdef.get('width', 0))
            h = float(pdef.get('height', 0))
            qty = int(pdef.get('quantity', 1))
            rot = bool(pdef.get('rotation_allowed', True))
            orig_file = pdef.get('originalFileName', None)

            for idx in range(1, qty + 1):
                inst_id = f"{name}-{idx:03d}"
                inst = PieceInstance(
                    instance_id=inst_id,
                    name=name,
                    original_file_name=orig_file,
                    width=w,
                    height=h,
                    rotation_allowed=rot,
                    alphabetical_rank=rank_counter
                )
                all_instances.append(inst)
                rank_counter += 1

        return all_instances


# ==============================================================================
# 3. GUILLOTINE CUTTING TREE ANALYZER
# ==============================================================================

class GuillotineCutNode:
    def __init__(self, region: Rectangle, pieces: list):
        self.region = region
        self.pieces = pieces
        self.cut_type = None  # 'V' or 'H'
        self.cut_position = None
        self.child1 = None
        self.child2 = None
        self.is_leaf = False


class GuillotineAnalyzer:
    @staticmethod
    def analyze_layout(sheet: Sheet, inside_pieces: list) -> dict:
        if not inside_pieces:
            return {
                'quality': 'FULL',
                'cut_count': 0,
                'total_length': 0.0,
                'direction_changes': 0,
                'tree': None,
                'locked': False
            }

        sheet_rect = sheet.to_rect()

        def solve_region(rect: Rectangle, pieces: list):
            if not pieces:
                return (True, 0, 0.0, [], None)

            if len(pieces) == 1:
                p = pieces[0]
                if abs(rect.x - p.x) < 1e-4 and abs(rect.y - p.y) < 1e-4 and \
                   abs(rect.w - p.effective_w) < 1e-4 and abs(rect.h - p.effective_h) < 1e-4:
                    node = GuillotineCutNode(rect, pieces)
                    node.is_leaf = True
                    return (True, 0, 0.0, [], node)

            candidate_cuts = []
            for p in pieces:
                for vx in [p.x, p.x2]:
                    if rect.x + 1e-4 < vx < rect.x2 - 1e-4:
                        candidate_cuts.append(('V', round(vx, 4)))
                for vy in [p.y, p.y2]:
                    if rect.y + 1e-4 < vy < rect.y2 - 1e-4:
                        candidate_cuts.append(('H', round(vy, 4)))

            candidate_cuts = list(set(candidate_cuts))
            best_sol = None

            for cut_type, cut_pos in candidate_cuts:
                is_valid_cut = True
                if cut_type == 'V':
                    left_pieces, right_pieces = [], []
                    for p in pieces:
                        pr = p.to_rect()
                        if pr.x < cut_pos < pr.x2:
                            is_valid_cut = False
                            break
                        if pr.x2 <= cut_pos:
                            left_pieces.append(p)
                        else:
                            right_pieces.append(p)

                    if not is_valid_cut:
                        continue

                    r_left = Rectangle(rect.x, rect.y, round(cut_pos - rect.x, 4), rect.h)
                    r_right = Rectangle(cut_pos, rect.y, round(rect.x2 - cut_pos, 4), rect.h)

                    sol1 = solve_region(r_left, left_pieces)
                    sol2 = solve_region(r_right, right_pieces)

                    if sol1[0] and sol2[0]:
                        total_cuts = 1 + sol1[1] + sol2[1]
                        length = round(rect.h + sol1[2] + sol2[2], 4)
                        dirs = ['V'] + sol1[3] + sol2[3]

                        node = GuillotineCutNode(rect, pieces)
                        node.cut_type = 'V'
                        node.cut_position = cut_pos
                        node.child1, node.child2 = sol1[4], sol2[4]

                        cost = (total_cuts, length)
                        if best_sol is None or cost < (best_sol[1], best_sol[2]):
                            best_sol = (True, total_cuts, length, dirs, node)

                elif cut_type == 'H':
                    top_pieces, bottom_pieces = [], []
                    for p in pieces:
                        pr = p.to_rect()
                        if pr.y < cut_pos < pr.y2:
                            is_valid_cut = False
                            break
                        if pr.y2 <= cut_pos:
                            bottom_pieces.append(p)
                        else:
                            top_pieces.append(p)

                    if not is_valid_cut:
                        continue

                    r_bottom = Rectangle(rect.x, rect.y, rect.w, round(cut_pos - rect.y, 4))
                    r_top = Rectangle(rect.x, cut_pos, rect.w, round(rect.y2 - cut_pos, 4))

                    sol1 = solve_region(r_bottom, bottom_pieces)
                    sol2 = solve_region(r_top, top_pieces)

                    if sol1[0] and sol2[0]:
                        total_cuts = 1 + sol1[1] + sol2[1]
                        length = round(rect.w + sol1[2] + sol2[2], 4)
                        dirs = ['H'] + sol1[3] + sol2[3]

                        node = GuillotineCutNode(rect, pieces)
                        node.cut_type = 'H'
                        node.cut_position = cut_pos
                        node.child1, node.child2 = sol1[4], sol2[4]

                        cost = (total_cuts, length)
                        if best_sol is None or cost < (best_sol[1], best_sol[2]):
                            best_sol = (True, total_cuts, length, dirs, node)

            if best_sol:
                return best_sol
            else:
                return (False, 0, 0.0, [], None)

        success, cut_count, total_length, dirs, tree = solve_region(sheet_rect, inside_pieces)

        direction_changes = 0
        for i in range(len(dirs) - 1):
            if dirs[i] != dirs[i + 1]:
                direction_changes += 1

        quality = 'FULL' if success else ('PARTIAL' if cut_count > 0 else 'NONE')

        return {
            'quality': quality,
            'cut_count': cut_count,
            'total_length': round(total_length, 4),
            'direction_changes': direction_changes,
            'tree': tree,
            'locked': not success
        }


# ==============================================================================
# 4. PACKING ENGINES (MAXRECTS & SKYLINE)
# ==============================================================================

class MaxRectsPacker:
    def __init__(self, sheet_w: float, sheet_h: float):
        self.sheet_w = sheet_w
        self.sheet_h = sheet_h
        self.free_rectangles = [Rectangle(0, 0, sheet_w, sheet_h)]

    def place_piece(self, p: PieceInstance, method: str = "BSSF") -> bool:
        orientations = []
        if p.rotation_allowed and p.original_w != p.original_h:
            orientations = [(p.original_w, p.original_h, False), (p.original_h, p.original_w, True)]
        else:
            orientations = [(p.original_w, p.original_h, False)]

        best_rect = None
        best_rot = False
        best_score1 = float('inf')
        best_score2 = float('inf')

        for pw, ph, rot in orientations:
            for fr in self.free_rectangles:
                if fr.w >= pw and fr.h >= ph:
                    leftover_w = abs(fr.w - pw)
                    leftover_h = abs(fr.h - ph)
                    short_side = min(leftover_w, leftover_h)
                    long_side = max(leftover_w, leftover_h)
                    area_fit = fr.w * fr.h - pw * ph

                    if method == "BSSF":
                        score1, score2 = short_side, long_side
                    elif method == "BLSF":
                        score1, score2 = long_side, short_side
                    elif method == "BAF":
                        score1, score2 = area_fit, short_side
                    elif method == "BL":
                        score1, score2 = fr.y + ph, fr.x + pw
                    else:
                        score1, score2 = short_side, long_side

                    if score1 < best_score1 or (score1 == best_score1 and score2 < best_score2):
                        best_score1 = score1
                        best_score2 = score2
                        best_rect = Rectangle(fr.x, fr.y, pw, ph)
                        best_rot = rot

        if best_rect is None:
            return False

        p.x = best_rect.x
        p.y = best_rect.y
        p.rotated = best_rot
        p.state = "INSIDE"

        self._split_free_rectangles(best_rect)
        self._prune_free_rectangles()
        return True

    def _split_free_rectangles(self, placed: Rectangle):
        new_free = []
        for fr in self.free_rectangles:
            if not fr.intersects_interior(placed):
                new_free.append(fr)
                continue

            if placed.y2 < fr.y2:
                new_free.append(Rectangle(fr.x, placed.y2, fr.w, round(fr.y2 - placed.y2, 4)))
            if placed.y > fr.y:
                new_free.append(Rectangle(fr.x, fr.y, fr.w, round(placed.y - fr.y, 4)))
            if placed.x > fr.x:
                new_free.append(Rectangle(fr.x, fr.y, round(placed.x - fr.x, 4), fr.h))
            if placed.x2 < fr.x2:
                new_free.append(Rectangle(placed.x2, fr.y, round(fr.x2 - placed.x2, 4), fr.h))

        self.free_rectangles = new_free

    def _prune_free_rectangles(self):
        pruned = []
        n = len(self.free_rectangles)
        for i in range(n):
            r1 = self.free_rectangles[i]
            is_contained = False
            for j in range(n):
                if i == j:
                    continue
                r2 = self.free_rectangles[j]
                if r2.contains(r1):
                    if r1.x == r2.x and r1.y == r2.y and r1.w == r2.w and r1.h == r2.h:
                        if i > j:
                            is_contained = True
                            break
                    else:
                        is_contained = True
                        break
            if not is_contained and r1.w > 0 and r1.h > 0:
                pruned.append(r1)
        self.free_rectangles = pruned


class SkylinePacker:
    def __init__(self, sheet_w: float, sheet_h: float):
        self.sheet_w = sheet_w
        self.sheet_h = sheet_h
        self.skyline = [[0.0, 0.0, sheet_w]]

    def place_piece(self, p: PieceInstance) -> bool:
        orientations = []
        if p.rotation_allowed and p.original_w != p.original_h:
            orientations = [(p.original_w, p.original_h, False), (p.original_h, p.original_w, True)]
        else:
            orientations = [(p.original_w, p.original_h, False)]

        best_y = float('inf')
        best_x = float('inf')
        best_index = -1
        best_rot = False
        best_pw, best_ph = 0, 0

        for pw, ph, rot in orientations:
            for i in range(len(self.skyline)):
                x = self.skyline[i][0]
                if x + pw > self.sheet_w + 1e-4:
                    continue

                y = 0.0
                j = i
                width_left = pw
                while width_left > 1e-4 and j < len(self.skyline):
                    y = max(y, self.skyline[j][1])
                    width_left -= self.skyline[j][2]
                    j += 1

                if y + ph <= self.sheet_h + 1e-4:
                    if y < best_y or (abs(y - best_y) < 1e-4 and x < best_x):
                        best_y = y
                        best_x = x
                        best_index = i
                        best_rot = rot
                        best_pw, best_ph = pw, ph

        if best_index == -1:
            return False

        p.x = best_x
        p.y = best_y
        p.rotated = best_rot
        p.state = "INSIDE"

        self._update_skyline(best_x, best_y, best_pw, best_ph)
        return True

    def _update_skyline(self, x: float, y: float, w: float, h: float):
        new_skyline = []
        new_y = round(y + h, 4)
        inserted = False

        for seg in self.skyline:
            sx, sy, sw = seg[0], seg[1], seg[2]
            sx2 = round(sx + sw, 4)
            px2 = round(x + w, 4)

            if sx2 <= x or sx >= px2:
                new_skyline.append(seg)
            else:
                if sx < x:
                    new_skyline.append([sx, sy, round(x - sx, 4)])
                if not inserted:
                    new_skyline.append([x, new_y, w])
                    inserted = True
                if sx2 > px2:
                    new_skyline.append([px2, sy, round(sx2 - px2, 4)])

        merged = []
        for seg in new_skyline:
            if not merged:
                merged.append(seg)
            else:
                prev = merged[-1]
                if abs(prev[1] - seg[1]) < 1e-4 and abs(prev[0] + prev[2] - seg[0]) < 1e-4:
                    prev[2] = round(prev[2] + seg[2], 4)
                else:
                    merged.append(seg)
        self.skyline = merged


# ==============================================================================
# 5. CANDIDATE GENERATOR & EVALUATORS
# ==============================================================================

class CandidateGenerator:
    @staticmethod
    def calculate_difficulty(p: PieceInstance, sheet: Sheet) -> float:
        area_ratio = p.area / sheet.area
        max_dim_ratio = max(p.original_w, p.original_h) / max(sheet.width, sheet.height)
        min_dim_ratio = min(p.original_w, p.original_h) / min(sheet.width, sheet.height)
        aspect = max(p.original_w, p.original_h) / max(min(p.original_w, p.original_h), 1e-4)
        return area_ratio * 4.0 + max_dim_ratio * 3.0 + min_dim_ratio * 2.0 + aspect * 0.5

    @classmethod
    def get_orderings(cls, instances: list, sheet: Sheet) -> list:
        orderings = [
            ('Area_Desc', sorted(instances, key=lambda p: p.area, reverse=True)),
            ('Area_Asc', sorted(instances, key=lambda p: p.area)),
            ('MaxDim_Desc', sorted(instances, key=lambda p: max(p.original_w, p.original_h), reverse=True)),
            ('MinDim_Desc', sorted(instances, key=lambda p: min(p.original_w, p.original_h), reverse=True)),
            ('Perimeter_Desc', sorted(instances, key=lambda p: p.original_w + p.original_h, reverse=True)),
            ('Aspect_Desc', sorted(instances, key=lambda p: max(p.original_w, p.original_h) / min(p.original_w, p.original_h), reverse=True)),
            ('Difficult_First', sorted(instances, key=lambda p: cls.calculate_difficulty(p, sheet), reverse=True)),
            ('Alpha_Asc', sorted(instances, key=lambda p: p.alphabetical_rank)),
            ('Alpha_Desc', sorted(instances, key=lambda p: p.alphabetical_rank, reverse=True)),
            ('Hybrid', sorted(instances, key=lambda p: (cls.calculate_difficulty(p, sheet), -p.alphabetical_rank), reverse=True))
        ]
        return orderings

    @classmethod
    def generate_candidates(cls, sheet: Sheet, instances: list) -> list:
        candidates = []
        orderings = cls.get_orderings(instances, sheet)
        methods = ["BSSF", "BLSF", "BAF", "BL"]

        for ord_name, ordered_pieces in orderings:
            for method in methods:
                cloned_pieces = [p.clone() for p in ordered_pieces]
                packer = MaxRectsPacker(sheet.width, sheet.height)
                for p in cloned_pieces:
                    packer.place_piece(p, method=method)
                candidates.append(cloned_pieces)

            cloned_pieces_sky = [p.clone() for p in ordered_pieces]
            sky_packer = SkylinePacker(sheet.width, sheet.height)
            for p in cloned_pieces_sky:
                sky_packer.place_piece(p)
            candidates.append(cloned_pieces_sky)

        return candidates


class LayoutValidator:
    @staticmethod
    def validate(sheet: Sheet, all_instances: list) -> tuple:
        sheet_rect = sheet.to_rect()
        inside_pieces = [p for p in all_instances if p.state == "INSIDE"]

        for p in inside_pieces:
            if p.effective_w <= 0 or p.effective_h <= 0:
                return False, f"Piece {p.id} has invalid dimensions."
            pr = p.to_rect()
            if not sheet_rect.contains(pr):
                return False, f"Piece {p.id} exceeds sheet boundaries."

        n = len(inside_pieces)
        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = inside_pieces[i], inside_pieces[j]
                if p1.to_rect().intersects_interior(p2.to_rect()):
                    return False, f"Overlap detected between {p1.id} and {p2.id}."

        instance_ids = [p.id for p in all_instances]
        if len(instance_ids) != len(set(instance_ids)):
            return False, "Duplicate piece instance detected."

        return True, "Valid"


class CandidateDeduplicator:
    @staticmethod
    def compute_signature(all_instances: list) -> str:
        inside_pieces = sorted([p for p in all_instances if p.state == "INSIDE"], key=lambda p: p.id)
        sig_parts = []
        for p in inside_pieces:
            sig_parts.append(f"{p.id}:{p.x}:{p.y}:{p.effective_w}:{p.effective_h}:{p.rotated}")
        return hashlib.md5("|".join(sig_parts).encode('utf-8')).hexdigest()

    @classmethod
    def deduplicate(cls, candidates: list) -> list:
        seen = set()
        unique_candidates = []
        for cand in candidates:
            sig = cls.compute_signature(cand)
            if sig not in seen:
                seen.add(sig)
                unique_candidates.append(cand)
        return unique_candidates


class LayoutEvaluator:
    @staticmethod
    def evaluate_metrics(sheet: Sheet, inside_pieces: list) -> dict:
        sheet_area = sheet.area
        placed_area = round(sum(p.area for p in inside_pieces), 4)
        geometric_waste = round(sheet_area - placed_area, 4)

        if not inside_pieces:
            return {
                'geometric_waste': geometric_waste,
                'placed_count': 0,
                'placed_area': 0.0,
                'compactness': 0.0,
                'alignment': 0.0,
                'fragmentation': 1.0
            }

        min_x = min(p.x for p in inside_pieces)
        min_y = min(p.y for p in inside_pieces)
        max_x = max(p.x2 for p in inside_pieces)
        max_y = max(p.y2 for p in inside_pieces)
        bbox_area = max(round((max_x - min_x) * (max_y - min_y), 4), 1e-4)
        compactness = round(placed_area / bbox_area, 4)

        shared_edge_len = 0.0
        n = len(inside_pieces)
        for i in range(n):
            p1 = inside_pieces[i]
            if abs(p1.x) < 1e-4 or abs(p1.x2 - sheet.width) < 1e-4:
                shared_edge_len += p1.effective_h
            if abs(p1.y) < 1e-4 or abs(p1.y2 - sheet.height) < 1e-4:
                shared_edge_len += p1.effective_w

            for j in range(i + 1, n):
                p2 = inside_pieces[j]
                if abs(p1.x2 - p2.x) < 1e-4 or abs(p2.x2 - p1.x) < 1e-4:
                    overlap_y1 = max(p1.y, p2.y)
                    overlap_y2 = min(p1.y2, p2.y2)
                    if overlap_y2 > overlap_y1:
                        shared_edge_len += (overlap_y2 - overlap_y1)
                if abs(p1.y2 - p2.y) < 1e-4 or abs(p2.y2 - p1.y) < 1e-4:
                    overlap_x1 = max(p1.x, p2.x)
                    overlap_x2 = min(p1.x2, p2.x2)
                    if overlap_x2 > overlap_x1:
                        shared_edge_len += (overlap_x2 - overlap_x1)

        alignment = round(shared_edge_len, 4)

        mr = MaxRectsPacker(sheet.width, sheet.height)
        for p in inside_pieces:
            mr._split_free_rectangles(p.to_rect())
            mr._prune_free_rectangles()
        fragmentation_index = len(mr.free_rectangles)

        return {
            'geometric_waste': geometric_waste,
            'placed_count': len(inside_pieces),
            'placed_area': placed_area,
            'compactness': compactness,
            'alignment': alignment,
            'fragmentation': fragmentation_index
        }


# ==============================================================================
# 6. LOCAL SEARCH & HIERARCHICAL RANKER
# ==============================================================================

class LocalSearchOptimizer:
    @staticmethod
    def compact(sheet: Sheet, instances: list) -> list:
        inside_pieces = [p for p in instances if p.state == "INSIDE"]
        outside_pieces = [p for p in instances if p.state == "OUTSIDE"]

        inside_pieces.sort(key=lambda p: (p.x, p.y))

        for p in inside_pieces:
            best_x = p.x
            test_p = p.clone()
            x_targets = {0.0}
            for other in inside_pieces:
                if other.id != p.id and other.x2 <= p.x:
                    x_targets.add(other.x2)

            for tx in sorted(x_targets, reverse=True):
                test_p.x = tx
                overlap = False
                for other in inside_pieces:
                    if other.id != p.id:
                        if test_p.to_rect().intersects_interior(other.to_rect()):
                            overlap = True
                            break
                if not overlap:
                    best_x = tx
                    break
            p.x = best_x

            best_y = p.y
            y_targets = {0.0}
            for other in inside_pieces:
                if other.id != p.id and other.y2 <= p.y:
                    y_targets.add(other.y2)

            for ty in sorted(y_targets, reverse=True):
                test_p.y = ty
                overlap = False
                for other in inside_pieces:
                    if other.id != p.id:
                        if test_p.to_rect().intersects_interior(other.to_rect()):
                            overlap = True
                            break
                if not overlap:
                    best_y = ty
                    break
            p.y = best_y

        return inside_pieces + outside_pieces

    @classmethod
    def optimize(cls, sheet: Sheet, instances: list) -> list:
        best_candidate = cls.compact(sheet, instances)
        outside_pieces = [p for p in best_candidate if p.state == "OUTSIDE"]
        if not outside_pieces:
            return best_candidate

        inside_pieces = [p for p in best_candidate if p.state == "INSIDE"]
        mr = MaxRectsPacker(sheet.width, sheet.height)
        for p in inside_pieces:
            mr._split_free_rectangles(p.to_rect())
            mr._prune_free_rectangles()

        outside_pieces.sort(key=lambda p: p.alphabetical_rank)
        for out_p in outside_pieces:
            if mr.place_piece(out_p, method="BSSF"):
                out_p.state = "INSIDE"

        return cls.compact(sheet, instances)


class LayoutCandidate:
    def __init__(self, sheet: Sheet, instances: list):
        self.instances = instances
        self.inside_pieces = [p for p in instances if p.state == "INSIDE"]
        self.outside_pieces = [p for p in instances if p.state == "OUTSIDE"]

        self.metrics = LayoutEvaluator.evaluate_metrics(sheet, self.inside_pieces)
        self.guillotine = GuillotineAnalyzer.analyze_layout(sheet, self.inside_pieces)
        self.signature = CandidateDeduplicator.compute_signature(instances)
        self.outside_alpha_score = sum(p.alphabetical_rank for p in self.outside_pieces)


class HierarchicalRanker:
    @staticmethod
    def rank_candidates(sheet: Sheet, candidates: list, tolerance: float = 1.0) -> LayoutCandidate:
        evaluated = [LayoutCandidate(sheet, cand) for cand in candidates]

        min_waste = min(cand.metrics['geometric_waste'] for cand in evaluated)

        near_optimal_pool = [
            cand for cand in evaluated
            if cand.metrics['geometric_waste'] <= min_waste + tolerance
        ]

        g_map = {'FULL': 3, 'PARTIAL': 2, 'NONE': 1}

        def sort_key(cand: LayoutCandidate):
            return (
                -cand.metrics['placed_count'],                   # PRIORITY 1: Max Placed Piece Count
                cand.metrics['geometric_waste'],                # PRIORITY 2: Minimum Waste
                -g_map[cand.guillotine['quality']],             # PRIORITY 3: Guillotine Quality
                cand.guillotine['cut_count'],                   # PRIORITY 4: Minimum Cut Count
                cand.guillotine['total_length'],                 # PRIORITY 5: Minimum Cut Length
                cand.guillotine['direction_changes'],           # PRIORITY 6: Min Direction Changes
                cand.metrics['fragmentation'],                 # PRIORITY 7: Minimum Waste Fragmentation
                -cand.metrics['alignment'],                     # PRIORITY 8: Maximum Alignment
                -cand.metrics['compactness'],                   # PRIORITY 9: Maximum Compactness
                -cand.outside_alpha_score,                      # PRIORITY 10: Alphabetical Outside Preference
                cand.signature                                  # PRIORITY 11: Deterministic Signature
            )

        near_optimal_pool.sort(key=sort_key)
        return near_optimal_pool[0]


# ==============================================================================
# 7. NESTING SOLVER CORE
# ==============================================================================

class NestingSolver:
    def __init__(self, sheet_width: float, sheet_height: float, tolerance: float = 1.0):
        self.sheet = Sheet(sheet_width, sheet_height)
        self.tolerance = tolerance

    def solve(self, pieces_def: list) -> dict:
        # Step 1: Reverse Alphabetical ranking & expansion
        instances = AlphabeticalManager.expand_and_rank_pieces(pieces_def)

        # Step 2: Multi-Strategy Candidate Generation
        raw_candidates = CandidateGenerator.generate_candidates(self.sheet, instances)

        # Step 3: Local Search & Compaction
        optimized_candidates = []
        for cand in raw_candidates:
            opt_cand = LocalSearchOptimizer.optimize(self.sheet, cand)
            optimized_candidates.append(opt_cand)

        # Step 4: Validation
        valid_candidates = []
        for cand in optimized_candidates:
            is_valid, msg = LayoutValidator.validate(self.sheet, cand)
            if is_valid:
                valid_candidates.append(cand)

        # Step 5: Deduplication
        unique_candidates = CandidateDeduplicator.deduplicate(valid_candidates)

        # Step 6: Hierarchical Lexicographic Ranking
        best_candidate = HierarchicalRanker.rank_candidates(
            self.sheet, unique_candidates, tolerance=self.tolerance
        )

        # Step 7: Position OUTSIDE Pieces outside sheet boundaries
        self._arrange_outside_pieces(best_candidate.instances)

        # Step 8: Explanation Generation
        explanation = self._generate_explanation(best_candidate)

        return {
            'best_candidate': best_candidate,
            'total_candidates_generated': len(raw_candidates),
            'valid_candidates_count': len(valid_candidates),
            'unique_candidates_count': len(unique_candidates),
            'explanation': explanation
        }

    def _arrange_outside_pieces(self, instances: list):
        outside_pieces = [p for p in instances if p.state == "OUTSIDE"]
        outside_pieces.sort(key=lambda p: p.alphabetical_rank)

        start_x = self.sheet.width + 20.0
        current_y = 0.0
        max_w_in_col = 0.0

        for p in outside_pieces:
            p.x = round(start_x, 4)
            p.y = round(current_y, 4)
            p.rotated = False
            p.state = "OUTSIDE"

            current_y += p.effective_h + 10.0
            max_w_in_col = max(max_w_in_col, p.effective_w)
            if current_y > self.sheet.height:
                current_y = 0.0
                start_x += max_w_in_col + 20.0
                max_w_in_col = 0.0

    def _generate_explanation(self, cand: LayoutCandidate) -> str:
        m = cand.metrics
        g = cand.guillotine
        ins_count = len(cand.inside_pieces)
        out_count = len(cand.outside_pieces)
        utilization = round((m['placed_area'] / self.sheet.area) * 100, 2)

        out_names = [p.name for p in cand.outside_pieces]

        lines = [
            "Why this layout?",
            f"1. Maximum parts placed: {ins_count} / {ins_count + out_count}",
            f"2. Lowest waste among top-part layouts: {m['geometric_waste']} mm² (Utilization: {utilization}%)",
            f"3. Guillotine Structure: {g['quality'].capitalize()}",
            f"4. Required cuts: {g['cut_count']}",
            f"5. Total cut length: {g['total_length']} mm",
            f"6. Direction changes: {g['direction_changes']}",
            f"7. Shared edges / Alignment: {m['alignment']} mm",
            f"8. Outside pieces: {', '.join(out_names) if out_names else 'None'}"
        ]
        return "\n".join(lines)


# ==============================================================================
# 8. FLASK APP & API ROUTES
# ==============================================================================

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/optimize', methods=['POST'])
def api_optimize():
    data = request.get_json() or {}
    
    # دریافت ابعاد بدون مقدار پیش‌فرض
    sheet_w_val = data.get('sheet_width')
    sheet_h_val = data.get('sheet_height')

    if sheet_w_val is None or sheet_h_val is None:
        return jsonify({"valid": False, "error": "لطفاً عرض و ارتفاع شیت را مشخص کنید."}), 400

    try:
        sheet_w = float(sheet_w_val)
        sheet_h = float(sheet_h_val)
    except (ValueError, TypeError):
        return jsonify({"valid": False, "error": "ابعاد شیت باید عدد معتبر باشند."}), 400

    pieces = data.get('pieces', [])

    if not pieces or sheet_w <= 0 or sheet_h <= 0:
        return jsonify({"valid": False, "error": "ورودی‌ها نامعتبر هستند یا لیست قطعات خالی است."}), 400

    solver = NestingSolver(sheet_width=sheet_w, sheet_height=sheet_h, tolerance=500.0)
    result = solver.solve(pieces)

    best_cand = result['best_candidate']
    placed_data = [p.to_dict() for p in best_cand.instances]

    return jsonify({
        "valid": True,
        "total_pieces": len(best_cand.instances),
        "placed_count": len(best_cand.inside_pieces),
        "sheet": {"width": sheet_w, "height": sheet_h},
        "best_layout": {
            "valid": True,
            "utilization": round((best_cand.metrics['placed_area'] / (sheet_w * sheet_h)) * 100.0, 2),
            "waste_area": best_cand.metrics['geometric_waste'],
            "placed": placed_data,
            "guillotine_quality": best_cand.guillotine['quality'],
            "cut_count": best_cand.guillotine['cut_count'],
            "total_cut_length": best_cand.guillotine['total_length'],
            "direction_change_count": best_cand.guillotine['direction_changes'],
            "shared_edge_length": best_cand.metrics['alignment'],
            "explanation": result['explanation']
        }
    })

@app.route('/api/validate', methods=['POST'])
def api_validate():
    data = request.get_json() or {}
    
    # دریافت ابعاد بدون مقدار پیش‌فرض
    sheet_w_val = data.get('sheet_width')
    sheet_h_val = data.get('sheet_height')

    if sheet_w_val is None or sheet_h_val is None:
        return jsonify({"valid": False, "error": "ابعاد شیت الزامی است."}), 400

    try:
        sheet_w = float(sheet_w_val)
        sheet_h = float(sheet_h_val)
    except (ValueError, TypeError):
        return jsonify({"valid": False, "error": "ابعاد شیت باید عدد معتبر باشند."}), 400

    placed_pieces = data.get('pieces', [])

    states = {}
    all_valid = True

    for p in placed_pieces:
        p_id = p.get('id')
        x = float(p.get('x', 0))
        y = float(p.get('y', 0))
        w = float(p.get('effective_w', p.get('width', 0)))
        h = float(p.get('effective_h', p.get('height', 0)))

        if x >= 0 and y >= 0 and (x + w) <= sheet_w + 1e-4 and (y + h) <= sheet_h + 1e-4:
            states[p_id] = "INSIDE"
        elif (x + w) <= 0 or x >= sheet_w or (y + h) <= 0 or y >= sheet_h:
            states[p_id] = "FULLY_OUTSIDE"
            all_valid = False
        else:
            states[p_id] = "PARTIALLY_OUTSIDE"
            all_valid = False

    return jsonify({
        "valid": all_valid,
        "states": states
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)