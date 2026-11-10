"""
select_roi.py — GUI tool for selecting the motion-intensive ROI of each camera (paper App. A).

Usage:
    python tools/select_roi.py

Features:
    1. Select a folder containing the video files of a scene
    2. Preview a frame of each video (any frame number can be shown)
    3. Drag/resize the ROI box on each video ("Sync All Boxes" moves all boxes together)
    4. "Default Setting": largest centered square, "Full Frame": no cropping
    5. Set the ViT input size
    6. Export roi.json into the video folder (a box outside the frame is moved inside it):
       {"input_size": 224, "rois": {"<video file name>": {"x", "y", "width", "height"}}}
"""

import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
from PIL import Image, ImageTk

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


def fit_roi_to_frame(roi: dict, frame_w: int, frame_h: int) -> dict:
    """Moves an ROI {x, y, width, height} inside a frame_w x frame_h frame.

    The size is kept unless it is larger than the frame, in which case it is
    reduced to the frame size.
    """
    width = min(max(roi["width"], 1), frame_w)
    height = min(max(roi["height"], 1), frame_h)
    x = min(max(roi["x"], 0), frame_w - width)
    y = min(max(roi["y"], 0), frame_h - height)
    return {"x": x, "y": y, "width": width, "height": height}


class RoiBox:
    """Draggable, resizable ROI box on a canvas."""

    HANDLE_SIZE = 8  # corner handle size in pixels

    def __init__(self, canvas: tk.Canvas, x: int, y: int, width: int, height: int, scale: float, on_change=None, square_mode_fn=None):
        self.canvas = canvas
        self.scale = scale  # display scale factor
        # Store in original image coordinates
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.on_change = on_change  # callback(action, dx, dy, new_width, new_height)
        self.square_mode_fn = square_mode_fn  # callable returning bool

        self._drag_data = {"action": None, "sx": 0, "sy": 0}

        # Draw rectangle
        self.rect_id = canvas.create_rectangle(
            *self._display_coords(),
            outline="lime",
            width=2,
        )
        # Draw transparent fill rectangle for easy dragging
        self.fill_id = canvas.create_rectangle(
            *self._display_coords(),
            fill="",
            outline="",
            stipple="gray12",
        )
        # Draw corner handle (bottom-right)
        self.handle_id = canvas.create_rectangle(
            *self._handle_coords(),
            fill="lime",
            outline="white",
            width=1,
        )

        # Bind events — both outline and fill area are draggable
        canvas.tag_bind(self.rect_id, "<ButtonPress-1>", self._on_press)
        canvas.tag_bind(self.rect_id, "<B1-Motion>", self._on_drag)
        canvas.tag_bind(self.fill_id, "<ButtonPress-1>", self._on_press)
        canvas.tag_bind(self.fill_id, "<B1-Motion>", self._on_drag)
        canvas.tag_bind(self.handle_id, "<ButtonPress-1>", self._on_handle_press)
        canvas.tag_bind(self.handle_id, "<B1-Motion>", self._on_handle_drag)

    def _display_coords(self):
        """Return display coordinates (scaled) for the rectangle."""
        s = self.scale
        return (self.x * s, self.y * s, (self.x + self.width) * s, (self.y + self.height) * s)

    def _handle_coords(self):
        """Return display coordinates (scaled) for the resize handle."""
        s = self.scale
        hs = self.HANDLE_SIZE
        bx = (self.x + self.width) * s
        by = (self.y + self.height) * s
        return (bx - hs, by - hs, bx + hs, by + hs)

    def update_display(self):
        self.canvas.coords(self.rect_id, *self._display_coords())
        self.canvas.coords(self.fill_id, *self._display_coords())
        self.canvas.coords(self.handle_id, *self._handle_coords())

    def _on_press(self, event):
        self._drag_data = {"action": "move", "sx": event.x, "sy": event.y}

    def _on_drag(self, event):
        dx = (event.x - self._drag_data["sx"]) / self.scale
        dy = (event.y - self._drag_data["sy"]) / self.scale
        self.x += dx
        self.y += dy
        self._drag_data["sx"] = event.x
        self._drag_data["sy"] = event.y
        self.update_display()
        if self.on_change:
            self.on_change("move", dx, dy, self.width, self.height)

    def apply_move(self, dx, dy):
        """Apply a move delta without triggering on_change (to avoid loops)."""
        self.x += dx
        self.y += dy
        self.update_display()

    def apply_resize(self, new_width, new_height):
        """Apply a new size without triggering on_change (to avoid loops)."""
        self.width = new_width
        self.height = new_height
        self.update_display()

    def _on_handle_press(self, event):
        self._drag_data = {"action": "resize", "sx": event.x, "sy": event.y}

    def _on_handle_drag(self, event):
        dx = (event.x - self._drag_data["sx"]) / self.scale
        dy = (event.y - self._drag_data["sy"]) / self.scale
        new_width = max(20, self.width + dx)
        new_height = max(20, self.height + dy)
        if self.square_mode_fn and self.square_mode_fn():
            side = max(new_width, new_height)
            new_width = side
            new_height = side
        self.width = new_width
        self.height = new_height
        self._drag_data["sx"] = event.x
        self._drag_data["sy"] = event.y
        self.update_display()
        if self.on_change:
            self.on_change("resize", 0, 0, new_width, new_height)

    def get_roi(self):
        return {
            "x": int(round(self.x)),
            "y": int(round(self.y)),
            "width": int(round(self.width)),
            "height": int(round(self.height)),
        }

    def set_roi(self, roi):
        """Sets the box to an ROI {x, y, width, height} in original image coordinates."""
        self.x, self.y = roi["x"], roi["y"]
        self.width, self.height = roi["width"], roi["height"]
        self.update_display()


class VideoPanel:
    """Panel for one video: frame display + ROI box + frame navigation."""

    DISPLAY_WIDTH = 400  # max display width per video

    def __init__(self, parent: tk.Frame, video_path: Path, on_box_change=None, square_mode_fn=None):
        self.video_path = video_path
        self.on_box_change = on_box_change  # callback(panel, action, dx, dy, new_size)
        self.square_mode_fn = square_mode_fn
        self.cap = cv2.VideoCapture(str(video_path))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.orig_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame_idx = 0

        # Scale to fit display
        self.scale = min(self.DISPLAY_WIDTH / self.orig_w, self.DISPLAY_WIDTH / self.orig_h)
        self.disp_w = int(self.orig_w * self.scale)
        self.disp_h = int(self.orig_h * self.scale)

        # Container
        self.frame = tk.LabelFrame(parent, text=video_path.name, padx=5, pady=5)

        # Canvas
        self.canvas = tk.Canvas(self.frame, width=self.disp_w, height=self.disp_h, bg="black")
        self.canvas.pack()

        # Frame navigation
        nav = tk.Frame(self.frame)
        nav.pack(fill=tk.X, pady=2)
        tk.Label(nav, text="Frame:").pack(side=tk.LEFT)
        self.frame_var = tk.StringVar(value="0")
        self.frame_entry = tk.Entry(nav, textvariable=self.frame_var, width=8)
        self.frame_entry.pack(side=tk.LEFT, padx=2)
        self.frame_entry.bind("<Return>", self._on_frame_change)
        tk.Label(nav, text=f"/ {self.total_frames - 1}").pack(side=tk.LEFT)
        tk.Button(nav, text="Go", command=self._on_frame_change).pack(side=tk.LEFT, padx=2)

        # Create default ROI box (centered, 50% of min dimension)
        default_size = min(self.orig_w, self.orig_h) // 2
        default_x = (self.orig_w - default_size) // 2
        default_y = (self.orig_h - default_size) // 2
        self.roi_box = RoiBox(
            self.canvas, default_x, default_y, default_size, default_size, self.scale,
            on_change=lambda action, dx, dy, nw, nh: (
                self.on_box_change(self, action, dx, dy, nw, nh) if self.on_box_change else None
            ),
            square_mode_fn=self.square_mode_fn,
        )

        # Show first frame
        self._show_frame(0)

    def _read_frame(self, idx: int):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _show_frame(self, idx: int):
        frame = self._read_frame(idx)
        if frame is None:
            return
        self.current_frame_idx = idx
        img = Image.fromarray(frame).resize((self.disp_w, self.disp_h), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("bg")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo, tags="bg")
        self.canvas.tag_lower("bg")
        # Re-raise ROI box elements
        self.canvas.tag_raise(self.roi_box.fill_id)
        self.canvas.tag_raise(self.roi_box.rect_id)
        self.canvas.tag_raise(self.roi_box.handle_id)

    def _on_frame_change(self, event=None):
        try:
            idx = int(self.frame_var.get())
            idx = max(0, min(idx, self.total_frames - 1))
            self.frame_var.set(str(idx))
            self._show_frame(idx)
        except ValueError:
            pass

    def get_roi(self):
        """ROI in pixels of the original video, moved inside the frame if needed."""
        return fit_roi_to_frame(self.roi_box.get_roi(), self.orig_w, self.orig_h)

    def destroy(self):
        self.cap.release()


class RoiSelectionApp:
    """Main application window."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ROI Selection Tool")
        self.panels: list[VideoPanel] = []
        self.video_folder: Path | None = None

        # Top bar
        top = tk.Frame(self.root, padx=10, pady=5)
        top.pack(fill=tk.X)

        tk.Button(top, text="Select Video Folder", command=self._select_folder).pack(side=tk.LEFT)
        self.folder_label = tk.Label(top, text="No folder selected", fg="gray")
        self.folder_label.pack(side=tk.LEFT, padx=10)

        tk.Label(top, text="ViT input size:").pack(side=tk.LEFT, padx=(20, 2))
        self.input_size_var = tk.StringVar(value="224")
        tk.Entry(top, textvariable=self.input_size_var, width=6).pack(side=tk.LEFT)

        self.sync_var = tk.BooleanVar(value=False)
        tk.Checkbutton(top, text="Sync All Boxes", variable=self.sync_var).pack(side=tk.LEFT, padx=(20, 0))
        self.square_var = tk.BooleanVar(value=False)
        tk.Checkbutton(top, text="Square Mode", variable=self.square_var).pack(side=tk.LEFT, padx=(8, 0))
        tk.Button(top, text="Default Setting", command=self._default_setting, bg="#2196F3", fg="white").pack(side=tk.LEFT, padx=(4, 0))
        tk.Button(top, text="Full Frame", command=self._full_frame, bg="#9C27B0", fg="white").pack(side=tk.LEFT, padx=(4, 0))

        tk.Button(top, text="Export JSON", command=self._export_json, bg="#4CAF50", fg="white").pack(
            side=tk.RIGHT
        )

        # Scrollable area for video panels
        container = tk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True)

        self.scroll_canvas = tk.Canvas(container)
        scrollbar = tk.Scrollbar(container, orient=tk.VERTICAL, command=self.scroll_canvas.yview)
        self.scrollable_frame = tk.Frame(self.scroll_canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all")),
        )
        self.scroll_canvas.create_window((0, 0), window=self.scrollable_frame, anchor=tk.NW)
        self.scroll_canvas.configure(yscrollcommand=scrollbar.set)

        self.scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _select_folder(self):
        folder = filedialog.askdirectory(title="Select Video Folder")
        if not folder:
            return

        self.video_folder = Path(folder)
        self.folder_label.config(text=str(self.video_folder), fg="black")

        # Clear existing panels
        for p in self.panels:
            p.destroy()
            p.frame.destroy()
        self.panels.clear()

        # Find video files
        video_files = sorted(
            f
            for f in self.video_folder.iterdir()
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        )

        if not video_files:
            messagebox.showwarning("No Videos", f"No video files found in:\n{folder}")
            return

        # Panels in a grid of up to 3 columns
        n = len(video_files)
        cols = 3 if n > 2 else n
        for i, vf in enumerate(video_files):
            panel = VideoPanel(self.scrollable_frame, vf, on_box_change=self._on_box_change, square_mode_fn=self.square_var.get)
            panel.frame.grid(row=i // cols, column=i % cols, padx=5, pady=5, sticky=tk.NW)
            self.panels.append(panel)

    def _on_box_change(self, source_panel, action, dx, dy, new_width, new_height):
        """Broadcast ROI box changes to all panels when sync is enabled."""
        if not self.sync_var.get():
            return
        for panel in self.panels:
            if panel is source_panel:
                continue
            if action == "move":
                panel.roi_box.apply_move(dx, dy)
            elif action == "resize":
                panel.roi_box.apply_resize(new_width, new_height)

    def _full_frame(self):
        """Set each ROI box to cover the entire frame of each camera."""
        if not self.panels:
            messagebox.showwarning("No Videos", "Please select a video folder first.")
            return

        for panel in self.panels:
            panel.roi_box.x = 0
            panel.roi_box.y = 0
            panel.roi_box.width = panel.orig_w
            panel.roi_box.height = panel.orig_h
            panel.roi_box.update_display()

    def _default_setting(self):
        """Set all ROI boxes to the largest centered square that fits all cameras."""
        if not self.panels:
            messagebox.showwarning("No Videos", "Please select a video folder first.")
            return

        # Use the smallest resolution across all cameras so every box fits
        min_w = min(p.orig_w for p in self.panels)
        min_h = min(p.orig_h for p in self.panels)
        size = min(min_w, min_h)

        for panel in self.panels:
            panel.roi_box.x = (panel.orig_w - size) / 2
            panel.roi_box.y = (panel.orig_h - size) / 2
            panel.roi_box.width = size
            panel.roi_box.height = size
            panel.roi_box.update_display()

    def _export_json(self):
        if not self.panels:
            messagebox.showwarning("No Videos", "Please select a video folder first.")
            return

        try:
            input_size = int(self.input_size_var.get())
        except ValueError:
            messagebox.showerror("Invalid input size", "ViT input size must be an integer.")
            return

        rois = {}
        for panel in self.panels:
            roi = panel.get_roi()
            panel.roi_box.set_roi(roi)  # show the box that is saved
            rois[panel.video_path.name] = roi
        data = {"input_size": input_size, "rois": rois}

        if self.video_folder is None:
            messagebox.showerror("Error", "No folder selected.")
            return
        out_path = self.video_folder / "roi.json"
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)

        messagebox.showinfo("Exported", f"Saved to:\n{out_path}")

    def run(self):
        self.root.mainloop()
        # Cleanup
        for p in self.panels:
            p.destroy()


if __name__ == "__main__":
    app = RoiSelectionApp()
    app.run()
