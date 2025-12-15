import { CSSProperties, ReactNode } from "react";
export type TagSize = "sm" | "md" | "lg";
export type TagVariant = "neutral" | "info" | "success" | "warning" | "danger";
export interface TagProps {
    className?: string;
    styles?: CSSProperties;
    icon?: ReactNode;
    text: ReactNode;
    size?: TagSize;
    variant?: TagVariant;
}
export declare const Tag: import("react").ForwardRefExoticComponent<TagProps & import("react").RefAttributes<HTMLDivElement>>;
//# sourceMappingURL=Tag.d.ts.map