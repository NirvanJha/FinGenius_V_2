import { CSSProperties, ReactElement } from "react";
import { Tag } from "../Tag";
export interface TagBlockProps {
    children: ReactElement<typeof Tag> | ReactElement<typeof Tag>[];
    styles?: CSSProperties;
    className?: string;
}
export declare const TagBlock: import("react").ForwardRefExoticComponent<TagBlockProps & import("react").RefAttributes<HTMLDivElement>>;
//# sourceMappingURL=TagBlock.d.ts.map