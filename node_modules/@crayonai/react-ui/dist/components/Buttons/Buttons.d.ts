import { CSSProperties, HTMLAttributes, ReactElement } from "react";
import { ButtonProps } from "../Button";
import { IconButtonProps } from "../IconButton";
type ButtonsVariant = "vertical" | "horizontal";
export interface ButtonsProps extends HTMLAttributes<HTMLDivElement> {
    variant?: ButtonsVariant;
    children: ReactElement<ButtonProps | IconButtonProps> | ReactElement<ButtonProps | IconButtonProps>[];
    className?: string;
    style?: CSSProperties;
}
export declare const Buttons: import("react").ForwardRefExoticComponent<ButtonsProps & import("react").RefAttributes<HTMLDivElement>>;
export {};
//# sourceMappingURL=Buttons.d.ts.map