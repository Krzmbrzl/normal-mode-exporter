#!/usr/bin/env python3

# Use of this source code is governed by a BSD-style license that can
# be found in the LICENSE file at the root of the source tree or at
# <https://github.com/Krzmbrzl/normal-mode-exporter/blob/main/LICENSE>

from typing import Optional
from typing import List
from typing import Tuple
from typing import TextIO

import argparse
import os
import sys
import re


class Atom:
    def __init__(self, element:str, coordinates: List[float]):
        element = element.strip()
        assert len(coordinates) == 3
        assert(element != "")

        self.element = element
        self.coordinates = coordinates


class Molecule:
    def __init__(self, atoms: List[Atom] = []):
        self.atoms = atoms

    def geometry(self) -> List[List[float]]:
        geometry = []

        for atom in self.atoms:
            geometry.append(atom.coordinates)

        return geometry


class NormalMode:
    def __init__(self, displacement_vec: List[List[float]], frequency: float, intensity: float, symmetry: str, ir_active: bool, raman_active: bool):
        self.displacement_vec = displacement_vec
        self.frequency = frequency
        self.intensity = intensity
        self.symmetry = symmetry
        self.ir_active = ir_active
        self.raman_active = raman_active


def main():
    parser = argparse.ArgumentParser(description="Export vibrational normal modes to easily accessible formats")

    parser.add_argument("--input", "-i", help="Path to the input file or directory", metavar="PATH", required=True)
    parser.add_argument("--output", "-o", help="Path to the output directory", metavar="PATH", required=True)
    parser.add_argument("--input-format", help="Specify the format of the provided input", choices=["auto", "turbomole"], default="auto")
    parser.add_argument("--step-size", help="The step size to use for animations", type=float, default=0.05, metavar="SIZE")
    parser.add_argument("--displacement-scaling", help="Scaling factor to apply to displacement vectors during animation", type=float, default=2,
        metavar="FACTOR")

    args = parser.parse_args()

    # sanitize paths
    args.input = os.path.realpath(args.input)
    args.output = os.path.realpath(args.output)

    initialize_io(input_path=args.input, output_path=args.output)
    args.input_format = detect_input_format(input_path=args.input, input_format=args.input_format)

    if args.input_format == "turbomole":
        molecule, normal_modes = process_turbomole_input(args.input)
    else:
        error("Unknown format \"%s\"" % args.input_format)

    export(args.output, molecule=molecule, normal_modes=normal_modes, step_size=args.step_size, scaling_factor=args.displacement_scaling)


def error(msg: str, exit_code: int = 1):
    """Print the given error and exit the program"""

    print("[ERROR]: ", msg, file=sys.stderr)
    sys.exit(exit_code)


def initialize_io(input_path: str, output_path: str):
    """Initialize the input and output paths. Performs basic verification and potentially creates output directory"""

    if not os.path.exists(input_path):
        error("Provided input path \"%s\" does not exist" % input_path)

    if os.path.exists(output_path):
        if not os.path.isdir(output_path):
            error("Provided output path \"%s\" is not a directory" % output_path)
    else:
        # Create output directory
        os.makedirs(output_path)


def detect_input_format(input_path: str, input_format: str) -> str:
    """Tries to figure out the format of the provided input"""

    input_format = input_format.lower()

    if not input_format == "auto":
        # Format explicitly given -> use that
        return input_format

    if os.path.isdir(input_path):
        if os.path.exists(os.path.join(input_path, "control")):
            # Presence of a control file indicates that TurboMole was used
            return "turbomole"

    # At this point we only support TurboMole inputs, so if we don't detect that,
    # we can't auto-detect the format.
    error("Unable to auto-detect input format. Please specify it explicitly.")


def process_turbomole_input(input_path: str) -> Tuple[Molecule, List[NormalMode]]:
    """Reads the necessary input from the provided TurboMole calculation directory"""

    if os.path.isfile(input_path):
        error("TurboMole format expects an input _directory_, but was provided with a file: \"%s\"" % input_path)

    control_file_lines = open(os.path.join(input_path, "control"), "r").read().split("\n")
    coordinate_content = ""
    normal_mode_content = ""
    vib_spectrum_content = ""

    # Find the relevant contents
    i = 0
    while i < len(control_file_lines):
        current_line = control_file_lines[i]
        if current_line.strip() == "$coord" or current_line.strip().startswith("$coord "):
            if "file=" in current_line:
                # Coordinates given as external file
                coordinate_content = open(os.path.join(input_path, current_line[current_line.find("file=") + len("file=") : ].strip()),"r").read()
                coordinate_content = coordinate_content.replace("$coord", "")

                # Git rid of any additional info in the coord file (e.g. internal coordinates)
                if "$" in coordinate_content:
                    coordinate_content = coordinate_content[ : coordinate_content.find("$")]

                coordinate_content = coordinate_content.strip()
            else:
                # Coordinates given in-place
                i += 1
                while i < len(control_file_lines) and not "$" in control_file_lines[i]:
                    coordinate_content += control_file_lines[i] + "\n"
                    i += 1
                coordinate_content = coordinate_content.strip()
        elif current_line.strip().startswith("$vibrational normal modes"):
            if not "file=" in current_line:
                error("Expected normal modes to be given in an external file")

            normal_mode_content = open(os.path.join(input_path, current_line[current_line.find("file=") + len("file=") : ].strip()), "r").read()
            normal_mode_content = normal_mode_content.replace("$vibrational normal modes", "").replace("$end", "").strip()
        elif current_line.strip().startswith("$vibrational spectrum"):
            if not "file=" in current_line:
                error("Expected vibrational spectrum to be given in an external file")

            vib_spectrum_content = open(os.path.join(input_path, current_line[current_line.find("file=") + len("file=") : ].strip())).read()
            vib_spectrum_content = vib_spectrum_content.replace("$vibrational spectrum", "").replace("$end", "")
            vib_spectrum_content = re.sub(r"^#.*\n", "", vib_spectrum_content, flags=re.MULTILINE).strip()

        i += 1


    if normal_mode_content == "":
        error("Unable to find TurboMole's normal modes")
    if coordinate_content == "":
        error("Unable to find equilibrium geometry")
    if vib_spectrum_content == "":
        error("Unable to find TurboMole's vibspectrum info")

    lines = vib_spectrum_content.split("\n")
    lines = [element.split() for element in lines]

    # Convert and filter columns in parsed vib spectrum
    vib_info_lines = []
    for elements in lines:
        # The original columns are modeNum, symmetry, frequency (in cm^{-1}), IR intensity, IR active, Raman active
        if len(elements) == 5:
            # Translations/rotations don't have a symmetry spec -> add a dummy one
            elements.insert(1, "-")
        assert len(elements) == 6
        vib_info_lines.append([elements[1], float(elements[2]), float(elements[3]), elements[4].lower() == "yes", elements[5].lower() == "yes"])

    symmetry_idx = 0
    frequency_idx = 1
    intensity_idx = 2
    ir_idx = 3
    raman_idx = 4

    # Turn coordinate_content into dummy XYZ format by prepending two empty lines
    coordinate_content = "\n\n" + coordinate_content
    molecule = parse_xyz(coordinate_content, 3, True)

    # Process normal modes
    displacements = []
    for line in normal_mode_content.split("\n"):
        for part in line.split():
            if "." in part:
                # All parts that don't contain a period are probably part of the indexing column,
                # which is just weird and therefore useless to us
                displacements.append(float(part))

    # We expect to read 3 displacements per atom in the molecule (3 cartesian coordinates) times the number
    # of normal modes, which is obtained via len(vib_info_lines)
    if not len(displacements) == len(molecule.atoms) * 3 * len(vib_info_lines):
        error("Expected to read %d displacements, but got %d" % (len(molecule.atoms) * 3 * len(vib_info_lines), len(displacements)))

    # The normal mode displacements first list the x-displacement for the first atom in the molecule for every mode, then the y-coordinate,
    # then z and then move on to the second atom and so on
    displacement_vectors = []
    i = 0
    while i < len(displacements):
        cartesian_idx = i % 3
        atom_idx = (i // 3) % len(molecule.atoms)
        mode_idx = (i // (3 * len(molecule.atoms))) % len(vib_info_lines)

        if len(displacement_vectors) == mode_idx:
            displacement_vectors.append([])
        if len(displacement_vectors[mode_idx]) == atom_idx:
            displacement_vectors[mode_idx].append([0, 0, 0])

        # print(i)
        # print("Mode ", mode_idx)
        # print("Atom ", atom_idx)
        # print("Coord ", cartesian_idx)

        displacement_vectors[mode_idx][atom_idx][cartesian_idx] = displacements[i]

        i += 1

    normal_modes = []
    for i in range(len(vib_info_lines)):
        normal_modes.append(NormalMode(displacement_vec=displacement_vectors[i], frequency=vib_info_lines[i][frequency_idx],
        intensity=vib_info_lines[i][intensity_idx], symmetry=vib_info_lines[i][symmetry_idx], ir_active=vib_info_lines[i][ir_idx],
        raman_active=vib_info_lines[i][raman_idx]))

    return molecule, normal_modes



def parse_xyz(content: str, element_col: int = 0, convert_bohr_to_ang: bool = False) -> Molecule:
    """Parses a Molecule object from a XYZ(-ish)-formatted input"""
    assert element_col < 4

    molecule = Molecule()

    skipped = 0

    x_col = 0 + (1 if element_col == 0 else 0)
    y_col = 1 + (1 if element_col == 1 else 0)
    z_col = 2 + (1 if element_col == 2 else 0)

    for line in content.split("\n"):
        # Skip leading atom count and comment lines
        if skipped < 2:
            skipped += 1
            continue

        parts = line.split()

        assert len(parts) == 4

        molecule.atoms.append(Atom(element=parts[element_col], coordinates=[float(parts[x_col]), float(parts[y_col]), float(parts[z_col])]))

        if convert_bohr_to_ang:
            molecule.atoms[-1].coordinates = [element * 0.52918 for element in molecule.atoms[-1].coordinates]

    return molecule


def export(output_path: str, molecule: Molecule, normal_modes: List[NormalMode], step_size: float = 0.05, scaling_factor: float = 2):
    """Exports the provided data into the output directory"""
    # Export static
    with open(os.path.join(output_path, "equilibrium_structure.xyz"), "w") as out_file:
        write_xyz(out_file, molecule)

    steps = int(1 / step_size)

    for i in range(len(normal_modes)):
        current_mode = normal_modes[i]

        # Export displacement vector for every mode
        with open(os.path.join(output_path, "mode_%05d.displacement" % (i + 1)), "w") as output_file:
            output_file.write("# Mode %d\n" % (i + 1))
            output_file.write("# freq=%.2fcm^{-1}, sym=%s, intensity=%.4f, IR: %s, Raman: %s\n" % (current_mode.frequency, current_mode.symmetry,
                current_mode.intensity, current_mode.ir_active, current_mode.raman_active))
            write_displacement(output_file, molecule, current_mode.displacement_vec)


        # Export animation in form of an XYZ file that contains multiple snapshots along the vibration
        with open(os.path.join(output_path, "mode_%05d.xyz" % (i + 1)), "w") as output_file:
            for k in range(-steps, steps + 1):
                factor = scaling_factor * step_size * k

                comment = "  Elongated along mode %d by a factor of %.6f" % (i + 1, factor)
                write_xyz(output_file, molecule, comment=comment, displacement=current_mode.displacement_vec, displacement_factor=factor)


def write_displacement(stream: TextIO, molecule: Molecule, displacement: List[List[float]]):
    """Write the given displacement into the given stream/file. The molecule object is needed to provide element names"""

    for i in range(len(displacement)):
        stream.write("%s %4.8f %4.8f %4.8f\n" % (molecule.atoms[i].element, displacement[i][0], displacement[i][1], displacement[i][2]))


def write_xyz(stream: TextIO, molecule: Molecule, displacement: Optional[List[List[float]]] = None, displacement_factor: float = 1, comment: str=""):
    """Write an XYZ format of the given molecule (optionally displaced) to the provided stream/file"""

    if not displacement is None:
        assert len(displacement) == len(molecule.atoms)

    stream.write("%d\n" % len(molecule.atoms))
    stream.write(comment.replace("\n", " ") + "\n")

    for i in range(len(molecule.atoms)):
        x = molecule.atoms[i].coordinates[0]
        y = molecule.atoms[i].coordinates[1]
        z = molecule.atoms[i].coordinates[2]

        if not displacement is None:
            x += displacement_factor * displacement[i][0]
            y += displacement_factor * displacement[i][1]
            z += displacement_factor * displacement[i][2]

        stream.write("%s %4.8f %4.8f %4.8f\n" % (molecule.atoms[i].element, x, y, z))


if __name__ == "__main__":
    main()
