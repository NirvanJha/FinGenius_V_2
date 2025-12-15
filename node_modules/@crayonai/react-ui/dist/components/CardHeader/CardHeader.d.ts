import { CSSProperties, ReactElement, ReactNode } from "react";
import { ButtonProps } from "../Button";
import { IconButtonProps } from "../IconButton";
export interface CardHeaderProps extends Omit<React.HTMLAttributes<HTMLDivElement>, "title"> {
    icon?: ReactNode;
    title?: ReactNode;
    subtitle?: ReactNode;
    actions?: ReactElement<ButtonProps | IconButtonProps> | ReactElement<ButtonProps | IconButtonProps>[];
    className?: string;
    styles?: CSSProperties;
}
export declare const CardHeader: import("react").ForwardRefExoticComponent<CardHeaderProps & import("react").RefAttributes<HTMLDivElement>>;
//# sourceMappingURL=CardHeader.d.ts.map