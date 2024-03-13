import math

landscape_areas = []  # Replace with your landscape areas
path = []
tiles = {}  # Tile types and their quantities
targets = {}  # Target counts for each bush type


def read_tiles_problem(file_path):
    landscape = []
    tiles = {}
    targets = {}
    solution = []

    section_number = 0

    with open(file_path, "r") as file:
        for line in file:
            line_striped = line.strip()
            if line_striped.startswith("#"):
                section_number += 1
                continue

            if section_number == 2:  # Landscape
                landscape.append(line)
            elif section_number == 3:  # Tiles
                pairs = line_striped.split(", ")
                for pair in pairs:
                    pair = pair.replace("{", "").replace("}", "")
                    if "=" in pair:
                        key, value = pair.split("=", maxsplit=1)
                        tiles[key.strip()] = int(value.strip())
            elif section_number == 4:  # Targets
                parts = line_striped.split(":", maxsplit=1)
                if len(parts) == 2:
                    target, count = parts
                    targets[int(target)] = int(count)

            elif section_number == 6:  # Solution
                parts = line_striped.split()
                if len(parts) == 3:
                    index, count, tile_type = parts
                    solution.append((int(index), int(count), tile_type))

    return {
        "landscape": landscape,
        "tiles": tiles,
        "targets": targets,
        "solution": solution,
    }


def convert_landscape_to_2d_array(landscape):
    max_length = max(((len(line) + 1) // 2 for line in landscape), default=0)

    landscape_2d = []
    for line in landscape:
        row = []
        for i, chr in enumerate(line):
            if i % 2 == 0:
                row.append(int(chr) if chr.isdigit() else 0)
        row.extend([0] * (max_length - len(row)))
        landscape_2d.append(row[:-1])
    return landscape_2d[:-1]


def split_into_4x4_blocks(landscape_2d):
    blocks = []
    rows = len(landscape_2d)
    cols = len(landscape_2d[0]) if rows > 0 else 0

    for i in range(0, rows, 4):
        for j in range(0, cols, 4):
            block = [row[j : j + 4] for row in landscape_2d[i : i + 4]]
            blocks.append(block)

    return blocks


def print_blocks(blocks):
    for block_index, block in enumerate(blocks):
        print(f"Block {block_index + 1}:")
        for row in block:
            print(" ".join(map(str, row)))
        print()


class CSP:

    def __init__(self):
        self.Solution = []
        self.TwoD_Land = []
        self.tiles = {}
        self.target = {}

    def read_tiles_problem(self, file_path):
        landscape = []
        tiles = {}
        targets = {}
        solution = []

        section_number = 0

        with open(file_path, "r") as file:
            maze = []
            for line in file:
                line_striped = line.strip()
                if line_striped.startswith("#"):
                    section_number += 1
                    continue

                if section_number == 2:  # Landscape
                    landscape.append(line)
                    ls = list(line)[:-1]
                    ls = list(map(lambda x: x.replace(" ", "0"), ls))
                    ls = list(map(lambda x: int(x), ls))
                    ls = [ele for idx, ele in enumerate(ls) if idx % 2 == 0]
                    maze.append(ls)

                elif section_number == 3:  # Tiles
                    pairs = line_striped.split(", ")
                    for pair in pairs:
                        pair = pair.replace("{", "").replace("}", "")
                        if "=" in pair:
                            key, value = pair.split("=", maxsplit=1)
                            tiles[key.strip()] = int(value.strip())
                elif section_number == 4:  # Targets
                    parts = line_striped.split(":", maxsplit=1)
                    if len(parts) == 2:
                        target, count = parts
                        targets[int(target)] = int(count)

                elif section_number == 6:  # Solution
                    parts = line_striped.split()
                    if len(parts) == 3:
                        index, count, tile_type = parts
                        solution.append((int(index), int(count), tile_type))
        print("maze prev:", landscape)
        print(maze)

        self.tiles = tiles
        self.target = targets

        # self.TwoD_Land = split_into_4x4_blocks(landscape)

    def ReadFile_F(self, file_name):
        maze = []
        with open(file_name, "r") as file:
            # getting the inital landscape
            for line in file:
                if "Landscape" in line:
                    break
            line_count = 0
            for line in file:
                line_count += 1
                ls = list(line)[:-1]
                ls = list(map(lambda x: x.replace(" ", "0"), ls))
                ls = list(map(lambda x: int(x), ls))
                ls = [ele for idx, ele in enumerate(ls) if idx % 2 == 0]
                maze.append(ls)
                if line_count > len(ls) - 1:
                    break
            for i in range(0, len(maze), 4):
                for j in range(0, len(maze[0]), 4):
                    area = [row[j : j + 4] for row in maze[i : i + 4]]
                    self.TwoD_Land.append(area)

        self.read_tiles_problem(file_name)
        print("TwoD_Land", self.TwoD_Land)
        print(type(self.TwoD_Land))
        print(type(self.TwoD_Land[0][0][0]))

    def Full_block(self, area):
        return {1: 0, 2: 0, 3: 0, 4: 0}

    def Outer_block(self, area):

        result = {bush_type: 0 for bush_type in range(1, 5)}

        central_portion = [row[1:3] for row in area[1:3]]

        for bush_type in result.keys():
            result[bush_type] = sum(row.count(bush_type) for row in central_portion)

        return result

    def calculate_L_shape(self, area, mode):
        # Initialize the result dictionary with bush types as keys and 0 as values.
        result = {bush_type: 0 for bush_type in range(1, 5)}

        # Depending on the mode, slice the area to get the relevant portion.
        if mode == 1:
            relevant_portion = [row[1:] for row in area[:-1]]
        elif mode == 2:
            relevant_portion = [row[1:] for row in area[1:]]
        elif mode == 3:
            relevant_portion = [row[:-1] for row in area[:-1]]
        elif mode == 4:
            relevant_portion = [row[:-1] for row in area[1:]]
        else:
            raise ValueError("Invalid mode for L-shape calculation.")

        # Sum the occurrences of each bush type in the relevant portion.
        for bush_type in result:
            result[bush_type] = sum(row.count(bush_type) for row in relevant_portion)

        return result

    # Smaller functions to call the main L_shape calculation with the specific mode.
    def L_shape_1(self, area):
        return self.calculate_L_shape(area, 1)

    def L_shape_2(self, area):
        return self.calculate_L_shape(area, 2)

    def L_shape_3(self, area):
        return self.calculate_L_shape(area, 3)

    def L_shape_4(self, area):
        return self.calculate_L_shape(area, 4)

    def apply(self, func, area):
        return func(area)

    def check(self, current_state, path):
        # 计算每种瓷砖使用的次数
        count_L_shapes = sum(path.count(f"L_shape_{i}") for i in range(1, 5))
        count_full_blocks = path.count("Full_block")
        count_outer_blocks = path.count("Outer_block")

        # 检查瓷砖使用量是否超标
        if any(
            [
                count_L_shapes > self.tiles["EL_SHAPE"],
                count_full_blocks > self.tiles["FULL_BLOCK"],
                count_outer_blocks > self.tiles["OUTER_BOUNDARY"],
            ]
        ):
            return "used too many tiles"

        # 计算与目标状态的差异
        differences = {
            key: self.target[key] - current_state.get(key, 0) for key in self.target
        }

        # 检查是否达到目标状态
        if all(diff == 0 for diff in differences.values()):
            return "end"

        # 检查是否有过多的灌木丛被揭露
        if any(diff < 0 for diff in differences.values()):
            return "too many visible bushes"

        # 未触发任何返回条件
        return None

    def csp_with_MRV(self, current_index, current_bush_counts, tile_functions):

        if current_index == len(self.TwoD_Land):

            return self.check(current_bush_counts, self.Solution) == "end"
        sorted_functions = sorted(
            tile_functions,
            key=lambda func: sum(
                value > 0 for value in func(self.TwoD_Land[current_index]).values()
            ),
        )

        for function in sorted_functions:

            self.Solution.append(function.__name__)

            area_result = self.apply(function, self.TwoD_Land[current_index])

            updated_counts = {
                bush: current_bush_counts.get(bush, 0) + area_result.get(bush, 0)
                for bush in area_result
            }

            validation_result = self.check(updated_counts, self.Solution)
            if validation_result in ["too many visible bushes", "used too many tiles"]:

                self.Solution.pop()
                continue

            if self.csp_with_MRV(current_index + 1, updated_counts, tile_functions):
                return True
            else:

                self.Solution.pop()

        return False

    def apply_path_to_landscape(self):

        def reset_matrix():
            return [[-1] * 4 for _ in range(4)]

        def copy_inner(source, dest, pattern):
            patterns = {
                "outer_block": ((1, -1), (1, -1)),
                "L_shape_1": ((None, -1), (1, None)),
                "L_shape_2": ((1, None), (1, None)),
                "L_shape_3": ((None, -1), (None, -1)),
                "L_shape_4": ((1, None), (None, -1)),
            }
            if pattern not in patterns:
                raise ValueError(f"Pattern {pattern} is not recognized.")

            row_slice, col_slice = patterns[pattern]
            for row_index, row in enumerate(source[row_slice[0] : row_slice[1]]):
                dest[row_index + (1 if row_slice[0] == 1 else 0)][
                    col_slice[0] : col_slice[1]
                ] = row[col_slice[0] : col_slice[1]]

        def calculate_bushes():

            bush_counts = {1: 0, 2: 0, 3: 0, 4: 0}
            for block in self.TwoD_Land:
                for row in block:
                    for bush_type in bush_counts.keys():
                        bush_counts[bush_type] += row.count(bush_type)
            return bush_counts

        for index, block_type in enumerate(self.Solution):
            temp = reset_matrix()

            if block_type == "Full_block":
                self.TwoD_Land[index] = temp
            elif block_type == "Outer_block":
                copy_inner(self.TwoD_Land[index], temp, "outer_block")
                self.TwoD_Land[index] = temp
            elif block_type in ["L_shape_1", "L_shape_2", "L_shape_3", "L_shape_4"]:
                copy_inner(self.TwoD_Land[index], temp, block_type)
                self.TwoD_Land[index] = temp
            else:
                raise ValueError("Invalid block type")

        return calculate_bushes()


def print_landscape_with_blocks(landscape_areas):

    def combine_matrices_horizontally(blocks):
        combined_result = ""
        for row_index in range(4):
            row_combined = ""
            for block in blocks:
                row_string = str(block[row_index]).strip("[]")
                row_string = row_string.replace(",", "")
                row_string = row_string.replace("-1", "0")
                row_combined += row_string + "  "
            combined_result += row_combined + "\n"
        return combined_result

    side_length = int(math.sqrt(len(landscape_areas)))
    combined_areas = ""
    for index in range(0, len(landscape_areas), side_length):
        horizontal_section = combine_matrices_horizontally(
            landscape_areas[index : index + side_length]
        )
        combined_areas += horizontal_section + "\n"
    print(combined_areas)


constraint_solver = CSP()
file_path = "tile1.txt"
problem = read_tiles_problem(file_path)
landscape = convert_landscape_to_2d_array(problem["landscape"])
landscape_split = split_into_4x4_blocks(landscape)

print("landscape", landscape_split)
for row in landscape:
    print(" ".join(map(str, row)))
# print_blocks(landscape_split)

tiles = problem["tiles"]
targets = problem["targets"]
print(tiles, targets)

tile_functions = [
    getattr(constraint_solver, "L_shape_" + str(i)) for i in range(1, 5)
] + [constraint_solver.Full_block, constraint_solver.Outer_block]


constraint_solver.ReadFile_F(file_path)


result = constraint_solver.csp_with_MRV(
    0, dict.fromkeys([1, 2, 3, 4], 0), tile_functions
)
if result:
    print(f"\nTarget: {constraint_solver.target}\n")
    print("Final Path:")
    print(constraint_solver.Solution, "\n")
    constraint_solver.apply_path_to_landscape()
    print_landscape_with_blocks(constraint_solver.TwoD_Land)
    # print_blocks(constraint_solver.landscape_areas)
else:
    print("No path found.\n")
