#!/usr/bin/env python3
"""
CarVision AI - Unified Master Entry Point
Run without arguments for an interactive console menu,
or pass CLI arguments directly (e.g. python run.py --mode damage-video).
"""

import sys
import os
import subprocess

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

def interactive_menu():
    while True:
        print("\n" + "=" * 60)
        print("🚗  CarVision AI: Vehicle Damage & Hazard Inspection System")
        print("=" * 60)
        print(" [1] Real-Time Video Damage Inspection (Live GUI + Slider)")
        print(" [2] Image Damage Inspection (Neon Polygon Contours)")
        print(" [3] Fire & Smoke Hazard Instance Segmentation")
        print(" [4] Full Video Batch Annotation (Export MP4)")
        print(" [5] Launch Web Dashboard (Streamlit UI in Browser)")
        print(" [6] Launch Desktop App (Tkinter Image Browser)")
        print(" [0] Exit")
        print("=" * 60)

        choice = input("Enter option [0-6]: ").strip()

        if choice == "1":
            print("\nStarting Real-Time Video Inspection...")
            from apps.live_gui import main as live_main
            live_main()

        elif choice == "2":
            print("\nRunning Image Damage Inspection...")
            from apps.cli import run_damage_image
            run_damage_image(
                source="data/images/download.jpg",
                conf=0.40,
                polygon_mode=True,
                output=None,
                no_show=False
            )

        elif choice == "3":
            print("\nRunning Fire Instance Segmentation...")
            from apps.cli import run_fire_image
            run_fire_image(
                source="data/images/download_2.jpg",
                conf=0.25,
                output=None,
                no_show=False
            )

        elif choice == "4":
            print("\nBatch Annotating Video (car.mp4)...")
            from apps.cli import run_annotate_video
            run_annotate_video(
                source="data/videos/car.mp4",
                conf=0.40,
                polygon_mode=False,
                output=None
            )

        elif choice == "5":
            print("\nLaunching Streamlit Web Dashboard at http://localhost:8501 ...")
            web_script = os.path.join(BASE_DIR, "apps", "web_dashboard.py")
            subprocess.run([sys.executable, "-m", "streamlit", "run", web_script])

        elif choice == "6":
            print("\nLaunching Desktop App...")
            from apps.desktop_app import main as desktop_main
            desktop_main()

        elif choice == "0":
            print("\nKhuda Hafiz! Exiting CarVision AI.")
            break

        else:
            print("\nInvalid choice. Please choose a number between 0 and 6.")

def main():
    if len(sys.argv) > 1:
        # Pass CLI arguments directly to the CLI module
        from apps.cli import main as cli_main
        cli_main()
    else:
        interactive_menu()

if __name__ == "__main__":
    main()
