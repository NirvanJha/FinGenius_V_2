import React from "react";
type AspectRatioType = "1:1" | "3:2" | "3:4" | "4:3" | "16:9";
type ScaleType = "fit" | "fill";
export interface ImageProps extends React.ImgHTMLAttributes<HTMLImageElement> {
    src: string;
    alt?: string;
    styles?: React.CSSProperties;
    className?: string;
    aspectRatio?: AspectRatioType;
    scale?: ScaleType;
}
export declare const Image: React.ForwardRefExoticComponent<ImageProps & React.RefAttributes<HTMLImageElement>>;
export {};
//# sourceMappingURL=Image.d.ts.map