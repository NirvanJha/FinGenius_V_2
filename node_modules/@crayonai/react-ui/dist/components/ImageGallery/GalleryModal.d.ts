import React from "react";
import { ImageItem } from "./ImageGallery";
export interface GalleryModalProps {
    images: ImageItem[];
    selectedImageIndex: number;
    setSelectedImageIndex: (index: number) => void;
    onClose: () => void;
}
export declare const GalleryModal: React.FC<GalleryModalProps>;
//# sourceMappingURL=GalleryModal.d.ts.map